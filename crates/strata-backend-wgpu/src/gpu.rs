//! The wgpu device and its WGSL kernels.
//!
//! Vulkan is tried first (every vendor on Windows and Linux, and the only
//! backend that covers AMD's RDNA well), then D3D12 on Windows and Metal on
//! macOS. No adapter at all is an error the caller turns into a CPU fallback.
//!
//! Scope, stated plainly: the MATMUL path runs on the device, because that is
//! where the time goes — one token touches hundreds of megabytes of expert
//! weights. The elementwise primitives (norm, rope, softmax, add) are a few
//! hundred floats each and stay on the host: a device round trip costs more
//! than the work. Weights are uploaded dequantized to f32 on first use and
//! evicted least-recently-used, because a 35 GB model does not fit in a 12 GB
//! card; the CPU path is unchanged for any tensor that is not resident.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use wgpu::util::DeviceExt;

use strata_core::backend::{ComputeBackend, QuantTensor, RopeCfg};
use strata_core::error::{Error, Result};
use strata_core::weights::ExpertView;
use strata_gguf::dequant::dequantize_row;

/// One WGSL kernel: `y[row] = dot(W[row], x)`.
const GEMV_WGSL: &str = r"
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    rows: u32,
    cols: u32,
    pad0: u32,
    pad1: u32,
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.rows) {
        return;
    }
    let base = row * params.cols;
    var acc = 0.0;
    for (var i = 0u; i < params.cols; i = i + 1u) {
        acc = acc + weights[base + i] * x[i];
    }
    y[row] = acc;
}
";

/// A weight matrix resident on the device.
struct Resident {
    buffer: wgpu::Buffer,
    bytes: u64,
    used: u64,
}

/// An LRU cache of uploaded weights, bounded by `budget` bytes.
struct WeightCache {
    entries: HashMap<String, Resident>,
    bytes: u64,
    budget: u64,
    clock: u64,
}

impl WeightCache {
    fn new(budget: u64) -> Self {
        Self {
            entries: HashMap::new(),
            bytes: 0,
            budget,
            clock: 0,
        }
    }

    fn get(&mut self, key: &str) -> Option<&Resident> {
        if let Some(r) = self.entries.get_mut(key) {
            self.clock += 1;
            r.used = self.clock;
            return Some(r);
        }
        None
    }

    fn insert(&mut self, key: String, buffer: wgpu::Buffer, bytes: u64) {
        // Evict least-recently-used until the new entry fits.
        while !self.entries.is_empty() && self.bytes + bytes > self.budget {
            let victim = self
                .entries
                .iter()
                .min_by_key(|(_, r)| r.used)
                .map(|(k, _)| k.clone())
                .expect("non-empty");
            if let Some(r) = self.entries.remove(&victim) {
                self.bytes -= r.bytes;
                tracing::debug!("wgpu: evicted {victim} ({} MiB resident)", self.bytes >> 20);
            }
        }
        self.clock += 1;
        self.bytes += bytes;
        self.entries.insert(
            key,
            Resident {
                buffer,
                bytes,
                used: self.clock,
            },
        );
    }
}

/// The wgpu backend.
pub struct WgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter_name: String,
    backend_name: &'static str,
    pipeline: wgpu::ComputePipeline,
    cache: Mutex<WeightCache>,
    /// Scratch buffers reused across calls, grown on demand.
    scratch: Mutex<Scratch>,
    cpu: strata_backend_cpu::CpuBackend,
}

struct Scratch {
    x: Option<wgpu::Buffer>,
    y: Option<wgpu::Buffer>,
    staging: Option<wgpu::Buffer>,
    params: Option<wgpu::Buffer>,
    x_cap: u64,
    y_cap: u64,
    staging_cap: u64,
}

impl Scratch {
    fn new() -> Self {
        Self {
            x: None,
            y: None,
            staging: None,
            params: None,
            x_cap: 0,
            y_cap: 0,
            staging_cap: 0,
        }
    }
}

impl std::fmt::Debug for WgpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuBackend")
            .field("adapter", &self.adapter_name)
            .field("backend", &self.backend_name)
            .finish_non_exhaustive()
    }
}

impl WgpuBackend {
    /// Probes for an adapter and builds the device. `max_weight_bytes` bounds
    /// the resident weights (a fraction of the card's memory, not all of it).
    pub fn probe_with_budget(max_weight_bytes: u64) -> Result<Self> {
        // Vulkan covers every vendor here; D3D12 and Metal are the fallbacks.
        let backends = wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|e| Error::Backend {
            message: format!("no GPU adapter ({e}); install a current Vulkan driver or use the CPU backend"),
        })?;
        let info = adapter.get_info();
        let backend_name = match info.backend {
            wgpu::Backend::Vulkan => "vulkan",
            wgpu::Backend::Dx12 => "dx12",
            wgpu::Backend::Metal => "metal",
            wgpu::Backend::Gl => "gl",
            _ => "other",
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("strata"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .map_err(|e| Error::Backend {
            message: format!("requesting the device failed: {e}"),
        })?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemv"),
            source: wgpu::ShaderSource::Wgsl(GEMV_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gemv"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_name: info.name,
            backend_name,
            pipeline,
            cache: Mutex::new(WeightCache::new(max_weight_bytes)),
            scratch: Mutex::new(Scratch::new()),
            cpu: strata_backend_cpu::CpuBackend::new(),
        })
    }

    /// Probes with a default budget of 4 GiB of resident weights.
    pub fn probe() -> Result<Self> {
        Self::probe_with_budget(4 << 30)
    }

    /// The adapter's name, for logs and `/v1/status`.
    pub fn adapter(&self) -> &str {
        &self.adapter_name
    }

    /// Bytes of resident weights.
    pub fn resident_bytes(&self) -> u64 {
        self.cache.lock().expect("cache").bytes
    }

    fn ensure_buffer(
        device: &wgpu::Device,
        cur: &mut Option<wgpu::Buffer>,
        cap: &mut u64,
        need: u64,
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> wgpu::Buffer {
        if cur.is_none() || *cap < need {
            let size = need.max(16).next_multiple_of(4);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            });
            *cap = size;
            *cur = Some(buf);
        }
        cur.as_ref().expect("buffer").clone()
    }
}

impl ComputeBackend for WgpuBackend {
    fn name(&self) -> &'static str {
        self.backend_name
    }

    fn gemv(&self, w: &QuantTensor<'_>, x: &[f32], y: &mut [f32], _: &mut [f32]) -> Result<()> {
        let key = format!("tensor:{}", w.name);
        self.run_gemv(&key, w.rows, w.cols, x, y, |out: &mut [f32]| {
            dequantize_row(w.quant, w.data, out).map_err(|e| Error::Backend {
                message: format!("dequantizing {}: {e}", w.name),
            })
        })
    }

    fn expert_gemv(
        &self,
        w: &ExpertView<'_>,
        e: usize,
        x: &[f32],
        y: &mut [f32],
        _: &mut [f32],
    ) -> Result<()> {
        let key = format!("expert:{}:{e}", w.name);
        let name = w.name;
        // An expert is stored `[in][out]`; the kernel wants `[out][in]`, so
        // the upload transposes it.
        let (rows_in, cols_out) = (w.rows, w.cols);
        self.run_gemv(&key, cols_out, rows_in, x, y, move |out: &mut [f32]| {
            let mut row = vec![0.0f32; cols_out];
            for i in 0..rows_in {
                dequantize_row(w.quant, w.expert_row(e, i), &mut row).map_err(|err| Error::Backend {
                    message: format!("dequantizing {name} expert {e} row {i}: {err}"),
                })?;
                for o in 0..cols_out {
                    out[o * rows_in + i] = row[o];
                }
            }
            Ok(())
        })
    }

    // ---- the elementwise primitives stay on the host; see the module docs.
    fn rms_norm(&self, x: &mut [f32], w: &[f32], eps: f64) -> Result<()> {
        self.cpu.rms_norm(x, w, eps)
    }

    fn rope_neox(&self, head: &mut [f32], cfg: &RopeCfg) -> Result<()> {
        self.cpu.rope_neox(head, cfg)
    }

    fn softmax(&self, xs: &mut [f32]) -> Result<()> {
        self.cpu.softmax(xs)
    }

    fn l2_norm(&self, head: &mut [f32], eps: f64) -> Result<()> {
        self.cpu.l2_norm(head, eps)
    }

    fn add(&self, a: &mut [f32], b: &[f32]) -> Result<()> {
        self.cpu.add(a, b)
    }
}

impl WgpuBackend {
    /// `y[out_rows] = W[out_rows][in_cols] · x[in_cols]`, with `W` uploaded in
    /// that `[out][in]` layout by `dequant`.
    fn run_gemv(
        &self,
        key: &str,
        out_rows: usize,
        in_cols: usize,
        x: &[f32],
        y: &mut [f32],
        dequant: impl FnOnce(&mut [f32]) -> Result<()>,
    ) -> Result<()> {
        let (rows, cols) = (out_rows, in_cols);
        assert_eq!(x.len(), cols, "gemv {key}: x");
        assert_eq!(y.len(), rows, "gemv {key}: y");
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let device = &self.device;
        // 1. the weight: upload dequantized f32 on first use.
        let weight_bytes = (rows * cols * 4) as u64;
        let buffer = {
            let mut cache = self.cache.lock().expect("cache");
            if let Some(r) = cache.get(key) {
                r.buffer.clone()
            } else {
                let mut host = vec![0.0f32; rows * cols];
                dequant(&mut host)?;
                let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(key),
                    contents: bytemuck::cast_slice(&host),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let out = buf.clone();
                cache.insert(key.to_string(), buf, weight_bytes);
                out
            }
        };

        // 2. x, y, params and the readback staging buffer.
        let mut guard = self.scratch.lock().expect("scratch");
        let sc = &mut *guard; // field-wise borrows, not a whole-guard one
        let x_buf = Self::ensure_buffer(
            device,
            &mut sc.x,
            &mut sc.x_cap,
            (cols * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            "x",
        );
        let y_buf = Self::ensure_buffer(
            device,
            &mut sc.y,
            &mut sc.y_cap,
            (rows * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "y",
        );
        let staging = Self::ensure_buffer(
            device,
            &mut sc.staging,
            &mut sc.staging_cap,
            (rows * 4) as u64,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            "staging",
        );
        if sc.params.is_none() {
            sc.params = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: &[0u8; 16],
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }));
        }
        let params_buf = sc.params.as_ref().expect("params").clone();
        drop(guard);

        self.queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(x));
        let params = [rows as u32, cols as u32, 0, 0];
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

        // 3. dispatch.
        let layout = self.pipeline.get_bind_group_layout(0);
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemv"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gemv") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemv"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let groups = rows.div_ceil(64);
            pass.dispatch_workgroups(groups as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&y_buf, 0, &staging, 0, (rows * 4) as u64);
        let submission = self.queue.submit(Some(enc.finish()));

        // 4. read the result back. Per-call synchronisation: correct, and the
        // first thing to remove once kernels are batched per layer.
        let slice = staging.slice(..(rows * 4) as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r.map_err(|e| e.to_string()));
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|e| Error::Backend {
                message: format!("waiting for the GPU failed: {e}"),
            })?;
        rx.recv()
            .map_err(|e| Error::Backend {
                message: format!("the readback never completed: {e}"),
            })?
            .map_err(|e| Error::Backend {
                message: format!("mapping the readback failed: {e}"),
            })?;
        {
            let view = slice.get_mapped_range().map_err(|e| Error::Backend {
                message: format!("reading the mapped buffer failed: {e}"),
            })?;
            let floats: &[f32] = bytemuck::cast_slice(&view);
            y.copy_from_slice(&floats[..rows]);
        }
        staging.unmap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strata_core::ops;
    use strata_gguf::quant::QuantType;

    /// Builds an f32 `rows x cols` matrix as GGUF bytes.
    fn f32_matrix(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> Vec<u8> {
        let mut v = Vec::with_capacity(rows * cols * 4);
        for r in 0..rows {
            for c in 0..cols {
                v.extend_from_slice(&f(r, c).to_le_bytes());
            }
        }
        v
    }

    /// The tests skip (loudly) when no adapter is available, so the suite
    /// stays green on a machine without a GPU.
    fn backend() -> Option<WgpuBackend> {
        match WgpuBackend::probe_with_budget(64 << 20) {
            Ok(b) => {
                eprintln!("wgpu test adapter: {} ({})", b.adapter(), b.name());
                Some(b)
            }
            Err(e) => {
                eprintln!("skipping the GPU test: {e}");
                None
            }
        }
    }

    #[test]
    fn gemv_matches_the_cpu_reference() {
        let Some(be) = backend() else {
            return;
        };
        let rows = 37; // not a multiple of the workgroup size: the tail matters
        let cols = 64;
        let data = f32_matrix(rows, cols, |r, c| ((r * 7 + c) % 11) as f32 * 0.25 - 1.0);
        let w = QuantTensor::new("t", QuantType::F32, rows, cols, &data);
        let x: Vec<f32> = (0..cols).map(|i| (i % 5) as f32 * 0.5 - 1.0).collect();
        let mut got = vec![0.0f32; rows];
        let mut scratch = vec![0.0f32; cols];
        be.gemv(&w, &x, &mut got, &mut scratch).expect("gpu gemv");

        let mut want = vec![0.0f32; rows];
        ops::gemv(&w, &x, &mut want, &mut scratch);
        for (i, (g, e)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - e).abs() < 1e-3,
                "row {i}: gpu {g} vs cpu {e} (tolerance 1e-3)"
            );
        }
    }

    #[test]
    fn expert_gemv_matches_the_cpu_reference() {
        let Some(be) = backend() else {
            return;
        };
        // Two experts of [rows = 5, cols = 8]: `ffn_*_exps`-shaped.
        let n_expert = 2;
        let rows = 5;
        let cols = 8;
        let mut data = Vec::new();
        for e in 0..n_expert {
            for r in 0..rows {
                for c in 0..cols {
                    let v = ((e * 100 + r * 10 + c) % 13) as f32 * 0.2 - 1.0;
                    data.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        let view = ExpertView {
            name: "exps",
            quant: QuantType::F32,
            n_expert,
            rows,
            cols,
            data: &data,
        };
        let x: Vec<f32> = (0..rows).map(|i| (i % 3) as f32 - 0.5).collect();
        let mut scratch = vec![0.0f32; cols.max(rows)];
        for e in 0..n_expert {
            let mut got = vec![0.0f32; cols];
            be.expert_gemv(&view, e, &x, &mut got, &mut scratch)
                .expect("gpu expert gemv");
            let mut want = vec![0.0f32; cols];
            ops::expert_gemv(&view, e, &x, &mut want, &mut scratch);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() < 1e-3, "expert {e} col {i}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn the_weight_cache_evicts_when_over_budget() {
        // A 24 MiB budget and three matrices of 10 MiB: the first must go.
        let Ok(be) = WgpuBackend::probe_with_budget(24 << 20) else {
            eprintln!("skipping the GPU test: no adapter");
            return;
        };
        let big = 2_500_000usize; // 10 MiB of f32
        let data = vec![0u8; big * 4];
        let x = [1.0f32];
        let mut y = vec![0.0f32; big];
        let mut scratch = vec![0.0f32; 1];
        for i in 0..3 {
            let name = format!("big{i}");
            let w = QuantTensor::new(&name, QuantType::F32, big, 1, &data);
            be.gemv(&w, &x, &mut y, &mut scratch).expect("upload");
        }
        assert!(
            be.resident_bytes() <= 2 * (big * 4) as u64,
            "the cache must stay inside its budget ({} bytes resident)",
            be.resident_bytes()
        );
    }
}
