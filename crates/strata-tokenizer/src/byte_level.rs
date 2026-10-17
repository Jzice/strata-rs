//! The GPT-2 byte-level BPE alphabet: every byte maps to a printable unicode
//! character, reversibly, so any UTF-8 input is representable with no UNK
//! token.
//!
//! Bytes 33..126, 161..172 and 174..255 map to themselves; the remaining 68
//! (space, newline, the C0 controls and the invisible high range) are shifted
//! into 256+n.

use std::collections::HashMap;
use std::sync::OnceLock;

fn tables() -> &'static (HashMap<u8, char>, HashMap<char, u8>) {
    static TABLES: OnceLock<(HashMap<u8, char>, HashMap<char, u8>)> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut bs: Vec<u16> = (0x21..0x7F).chain(0xA1..0xAD).chain(0xAE..0x100).collect();
        let mut cs: Vec<u16> = bs.clone();
        let mut n: u16 = 0;
        for b in 0..=255u16 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut byte_to_unicode = HashMap::with_capacity(256);
        let mut unicode_to_byte = HashMap::with_capacity(256);
        for (b, c) in bs.into_iter().zip(cs) {
            let ch = char::from_u32(u32::from(c)).expect("valid unicode scalar");
            byte_to_unicode.insert(b as u8, ch);
            unicode_to_byte.insert(ch, b as u8);
        }
        (byte_to_unicode, unicode_to_byte)
    })
}

/// The byte's character in the byte-level alphabet.
pub fn byte_to_char(b: u8) -> char {
    tables().0[&b]
}

/// The alphabet character's byte, or `None` when it is outside the alphabet.
pub fn char_to_byte(c: char) -> Option<u8> {
    tables().1.get(&c).copied()
}

/// Maps a piece's UTF-8 bytes into the byte-level alphabet string.
pub fn encode_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| byte_to_char(b)).collect()
}

/// Maps a token's alphabet string back into its raw bytes. A character
/// outside the byte alphabet makes the whole token invalid.
pub fn decode_token(token: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(token.len());
    for c in token.chars() {
        out.push(char_to_byte(c)?);
    }
    Some(out)
}
