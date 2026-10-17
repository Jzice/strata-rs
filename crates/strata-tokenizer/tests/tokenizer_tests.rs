//! Tokenizer tests: the byte-level alphabet, the pre-tokenizer splits, BPE
//! merge ordering (rank order, not list order), the special-token classes,
//! the heap path, and the decisive round trip
//! `decode(encode(s)) == s` over a corpus that hits the boundaries.

use strata_tokenizer::bpe::Merges;
use strata_tokenizer::byte_level::{byte_to_char, char_to_byte, decode_token, encode_bytes};
use strata_tokenizer::pretok::{PreTokenizer, QWEN35_PATTERN};
use strata_tokenizer::tokenizer::{TokenType, Tokenizer};

/// A vocabulary with every byte-alphabet character plus a few merges and two
/// special tokens (one type 3, one type 4).
fn test_tokenizer() -> Tokenizer {
    let mut tokens: Vec<String> = (0..=255u16).map(|b| byte_to_char(b as u8).to_string()).collect();
    // The merged results must be vocabulary entries for the lookup to find
    // them (as in a real vocabulary).
    tokens.push("he".to_string());
    tokens.push("Ġl".to_string());
    let merges = vec!["h e".to_string(), "Ġ l".to_string()];
    let mut token_types = vec![TokenType::Normal; tokens.len()];
    tokens.push("<|special_a|>".to_string());
    token_types.push(TokenType::Control);
    tokens.push("<|special_b|>".to_string());
    token_types.push(TokenType::UserDefined);
    Tokenizer::new(tokens, &merges, token_types, "qwen35".to_string()).expect("tokenizer")
}

#[test]
fn byte_alphabet_is_a_bijection() {
    // 256 bytes map to 256 distinct printable characters, reversibly.
    let mut seen = std::collections::HashSet::new();
    for b in 0..=255u8 {
        let c = byte_to_char(b);
        assert!(seen.insert(c), "byte {b} maps to a repeated character {c:?}");
        assert_eq!(char_to_byte(c), Some(b));
        assert_eq!(decode_token(&c.to_string()), Some(vec![b]));
    }
    // The classic GPT-2 anchors: space -> 'Ġ', newline -> 'Ċ'.
    assert_eq!(byte_to_char(b' '), 'Ġ');
    assert_eq!(byte_to_char(b'\n'), 'Ċ');
}

#[test]
fn encode_bytes_maps_every_byte() {
    let s = encode_bytes(&[0x00, 0x20, 0xE4, 0xBD, 0xA0]);
    assert_eq!(decode_token(&s), Some(vec![0x00, 0x20, 0xE4, 0xBD, 0xA0]));
}

#[test]
fn pretokenizer_splits_qwen35_pieces() {
    let pt = PreTokenizer::qwen35().unwrap();
    let joined = |s: &str| pt.split(s).concat();
    // The pattern is total: the pieces concatenate back to the input.
    let long = "x".repeat(200);
    for s in [
        "hello world",
        "  leading and trailing  ",
        "a\n\n\nb",
        "def f(x):\n\treturn x  # comment\n",
        "你好，世界",
        "😀🚀",
        "e\u{301}\u{301} combining",
        "\r\n\r\n",
        "it's I'll we've",
        "1234567890",
        long.as_str(),
    ] {
        assert_eq!(joined(s), s, "pieces must cover the input exactly");
    }
    // Spot checks of the boundaries that matter.
    assert_eq!(pt.split("hello world"), vec!["hello", " world"]);
    assert_eq!(pt.split("a\n\n\nb"), vec!["a", "\n\n\n", "b"]);
    assert_eq!(pt.split("it's"), vec!["it", "'s"]);
    // A leading punctuation character joins the letter run that follows it
    // (the pattern's optional `[^\r\n\p{L}\p{N}]?` prefix).
    assert_eq!(pt.split("你好，世界"), vec!["你好", "，世界"]);
    // The \p{M} cases: a combining mark stays with its letter.
    assert_eq!(pt.split("e\u{301}"), vec!["e\u{301}"]);
}

#[test]
fn bpe_merges_by_rank_not_list_order() {
    // "ab" with rank 0 and "a" "b" singles: the lowest rank first.
    let tokens: [String; 3] = ["a".to_string(), "b".to_string(), "ab".to_string()];
    let merges = Merges::from_pairs(["a b".to_string()]);
    let vocab: std::collections::HashMap<String, u32> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();
    let pieces = merges.apply("ab");
    assert_eq!(pieces, vec!["ab"]);
    assert_eq!(vocab[&pieces[0]], 2);

    // Two overlapping merges: the lower rank wins regardless of position.
    let merges = Merges::from_pairs(["b c".to_string(), "a b".to_string()]);
    // "b c" has rank 0, so it merges first even though it is to the right.
    assert_eq!(merges.apply("abc"), vec!["a", "bc"]);
    let merges = Merges::from_pairs(["a b".to_string(), "b c".to_string()]);
    assert_eq!(merges.apply("abc"), vec!["ab", "c"]);
}

#[test]
fn bpe_heap_matches_scan() {
    // A long word (past HEAP_MIN) must merge exactly as the plain scan does.
    let alphabet: Vec<String> = (0..=255u16).map(|b| byte_to_char(b as u8).to_string()).collect();
    let mut tokens = alphabet.clone();
    let mut merges_list = Vec::new();
    // Chain merges: "a a" -> "aa", "aa a" -> "aaa", ... every rank used once.
    let mut acc = "a".to_string();
    for _ in 0..80 {
        let merged = format!("{acc}a");
        tokens.push(merged.clone());
        merges_list.push(format!("{acc} a"));
        acc = merged;
    }
    let merges = Merges::from_pairs(&merges_list);
    let word: String = "a".repeat(90);
    let scan = merges.apply_scan(&word);
    let heap = merges.apply_heap(&word);
    assert_eq!(scan, heap);
    // The scan and heap paths agree with each other on a mixed word too.
    let word2 = format!("{}{}", "a".repeat(70), "z".repeat(70));
    assert_eq!(merges.apply_scan(&word2), merges.apply_heap(&word2));
}

#[test]
fn special_token_classes() {
    let mut tk = test_tokenizer();
    let control = tk.token_id("<|special_a|>").unwrap();
    let user = tk.token_id("<|special_b|>").unwrap();

    // Type 3 (CONTROL): matched only with parse_special.
    let text = "a<|special_a|>b";
    let ids = tk.encode(text, false).unwrap();
    assert!(!ids.contains(&control), "type 3 needs parse_special");
    let ids = tk.encode(text, true).unwrap();
    assert!(ids.contains(&control));

    // Type 4 (USER_DEFINED): matched either way.
    let ids = tk.encode("a<|special_b|>b", false).unwrap();
    assert!(ids.contains(&user), "type 4 is matched always");
}

#[test]
fn round_trip_over_the_boundary_corpus() {
    let mut tk = test_tokenizer();
    let long = "x".repeat(5000);
    let corpus: [&str; 16] = [
        "",
        "Hello, world!",
        "  leading and trailing  ",
        "a\n\n\nb",
        "def f(x):\n\treturn x  # comment\n",
        "你好，世界",
        "مرحبا",
        "😀🚀🇺🇸",
        "e\u{301}\u{301} combining",
        "\x00\x01\x7f control",
        "non-breaking\u{a0}space",
        "1234567890",
        "MixedCASE_and-dashes",
        "\r\n\r\n",
        "\u{2028}\u{2029}",
        &long,
    ];
    let mut failed = 0;
    for s in corpus {
        let ids = tk.encode(s, false).expect("encode");
        let back = tk.decode(&ids);
        if back != s {
            println!("ROUND TRIP FAILED {s:?} -> {ids:?} -> {back:?}");
            failed += 1;
        }
    }
    assert_eq!(failed, 0, "{failed} strings failed the round trip");
}

#[test]
fn decode_replaces_invalid_utf8_lossily() {
    let tk = test_tokenizer();
    // A token id outside the vocabulary decodes to U+FFFD, not a panic.
    let out = tk.decode(&[u32::MAX]);
    assert_eq!(out, "\u{FFFD}");
}

#[test]
fn qwen35_pattern_is_the_shipped_one() {
    // The pattern is shipped, not re-derived: pin its source string.
    assert!(QWEN35_PATTERN.contains(r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"));
    assert!(QWEN35_PATTERN.contains(r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"));
    assert!(QWEN35_PATTERN.contains(r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"));
    assert!(QWEN35_PATTERN.contains(r"\s*[\r\n]+"));
    assert!(QWEN35_PATTERN.contains(r"\s+(?!\S)"));
    assert!(QWEN35_PATTERN.ends_with(r"|\s+"));
}

#[test]
fn chat_template_renders_the_conversation() {
    use strata_tokenizer::template::ChatTemplate;
    // A Qwen-style template with the loop and the generation prompt.
    let source = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant:\n{% endif %}";
    let tpl = ChatTemplate::new(source);
    let out = tpl
        .render(
            &[
                strata_tokenizer::template::Message {
                    role: "system".into(),
                    content: "You are Strata.".into(),
                },
                strata_tokenizer::template::Message {
                    role: "user".into(),
                    content: "Hi".into(),
                },
            ],
            true,
        )
        .unwrap();
    assert_eq!(out, "system: You are Strata.\nuser: Hi\nassistant:\n");
}
