//! # strata-backend-wgpu
//!
//! The wgpu implementation of [`strata_core::backend::ComputeBackend`]: one
//! set of WGSL kernels for every vendor (Vulkan first on Windows/Linux, D3D12
//! as the fallback), weight repacking for GPU-friendly layouts, and a switch
//! back to the CPU backend when no suitable adapter exists.
//!
//! The whole GPU path is behind the `gpu` feature; without it the crate only
//! provides backend selection.

use strata_core::backend::ComputeBackend;
use strata_core::error::Result;

/// Which backend the engine should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendChoice {
    /// Pick the best available: GPU when a suitable adapter exists, else CPU.
    #[default]
    Auto,
    /// Require the GPU (error when no adapter exists).
    Gpu,
    /// Force the CPU backend.
    Cpu,
}

/// Selects and constructs a backend. With the `gpu` feature this probes
/// Vulkan first, then D3D12 (Windows) / Metal (macOS); without it, always CPU.
pub fn select(choice: BackendChoice) -> Result<Box<dyn ComputeBackend>> {
    match choice {
        BackendChoice::Cpu => Ok(Box::new(strata_backend_cpu::CpuBackend::new())),
        BackendChoice::Auto | BackendChoice::Gpu => {
            #[cfg(feature = "gpu")]
            {
                match WgpuBackend::probe() {
                    Ok(backend) => {
                        tracing::info!("wgpu backend: {}", backend.name());
                        Ok(Box::new(backend))
                    }
                    Err(e) => match choice {
                        BackendChoice::Auto => {
                            tracing::warn!("no usable GPU ({e}); falling back to the CPU backend");
                            Ok(Box::new(strata_backend_cpu::CpuBackend::new()))
                        }
                        BackendChoice::Gpu => Err(strata_core::error::Error::Backend {
                            message: format!("no usable GPU: {e}"),
                        }),
                        BackendChoice::Cpu => unreachable!(),
                    },
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                let _ = choice;
                Ok(Box::new(strata_backend_cpu::CpuBackend::new()))
            }
        }
    }
}

#[cfg(feature = "gpu")]
mod gpu;

#[cfg(feature = "gpu")]
pub use gpu::WgpuBackend;
