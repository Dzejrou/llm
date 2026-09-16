//! Behavioural tests for the public tokenizer API.
//!
//! These exercise the crate exactly as a user does — through its public
//! interface — so they also serve as worked examples of the encode/decode
//! contract, in particular of which inputs are expected to fail and how.

use std::error::Error;

use tiny_llm::{ByteTokenizer, DecodeError, TokenId};

/// Encodes `text` and decodes the result, asserting the text survives intact.
///
/// A tokenizer that loses information here would silently corrupt every piece
/// of training data fed through it, so this round trip is the property the
/// whole module exists to guarantee.
fn assert_round_trips(text: &str) {
    let tokenizer = ByteTokenizer::new();
    let ids = tokenizer.encode(text);
    assert_eq!(
        tokenizer.decode(&ids).as_deref(),
        Ok(text),
        "round trip failed for {text:?}"
    );
}

#[test]
fn vocabulary_covers_exactly_the_byte_values() {
    assert_eq!(ByteTokenizer::VOCAB_SIZE, 256);
    assert_eq!(ByteTokenizer::new().vocab_size(), 256);
}

#[test]
fn empty_input_round_trips() {
    let tokenizer = ByteTokenizer::new();
    assert!(tokenizer.encode("").is_empty());
    assert_eq!(tokenizer.decode(&[]), Ok(String::new()));
    assert_round_trips("");
}

#[test]
fn ascii_encodes_to_its_byte_values() {
    assert_eq!(ByteTokenizer::new().encode("Hi"), vec![72, 105]);
    assert_round_trips("Hi");
}

#[test]
fn multibyte_characters_encode_to_several_tokens() {
    let tokenizer = ByteTokenizer::new();

    // U+00E9 LATIN SMALL LETTER E WITH ACUTE is two bytes in UTF-8.
    assert_eq!(tokenizer.encode("é"), vec![195, 169]);
    // U+1F980 CRAB is four.
    assert_eq!(tokenizer.encode("🦀"), vec![240, 159, 166, 128]);

    assert_round_trips("é");
    assert_round_trips("🦀");
    assert_round_trips("Příliš žluťoučký kůň úpěl ďábelské ódy 🦀");
}

#[test]
fn token_count_can_exceed_character_count() {
    let text = "héllo 🦀";
    let ids = ByteTokenizer::new().encode(text);

    // Seven characters, but 'é' costs two bytes and the crab costs four, so
    // the model would see eleven positions of context, not seven.
    assert_eq!(text.chars().count(), 7);
    assert_eq!(ids.len(), 11);
    assert_eq!(ids.len(), text.len());
}

#[test]
fn whitespace_newlines_and_nul_survive_unchanged() {
    let text = "a\tb\r\nc \n\n  d\0e\0";
    let tokenizer = ByteTokenizer::new();
    let ids = tokenizer.encode(text);

    // Nothing is trimmed, collapsed, or treated as a terminator: an embedded
    // NUL is just token 0.
    assert_eq!(ids.iter().filter(|&&id| id == 0).count(), 2);
    assert_round_trips(text);
}

#[test]
fn every_ascii_byte_including_control_codes_decodes() {
    let tokenizer = ByteTokenizer::new();

    // 0..=127 is exactly the range that is valid UTF-8 one byte at a time, so
    // each such ID must decode on its own and the whole run must decode
    // together.
    for id in 0..=127u32 {
        let decoded = tokenizer
            .decode(&[id])
            .unwrap_or_else(|error| panic!("id {id} should decode: {error}"));
        let mut chars = decoded.chars();
        assert_eq!(chars.next().map(u32::from), Some(id));
        assert!(chars.next().is_none());
    }

    let all: Vec<TokenId> = (0..=127).collect();
    let decoded = tokenizer.decode(&all).expect("ASCII run should decode");
    assert_eq!(decoded.len(), 128);
    assert_eq!(tokenizer.encode(&decoded), all);
}

#[test]
fn ids_above_the_vocabulary_are_rejected_with_their_position() {
    let tokenizer = ByteTokenizer::new();

    // 256 is the first ID that names no byte; truncating the cast would have
    // turned it into byte 0 and produced a plausible-looking NUL.
    assert_eq!(
        tokenizer.decode(&[256]),
        Err(DecodeError::OutOfVocabulary {
            position: 0,
            id: 256
        })
    );
    assert_eq!(
        tokenizer.decode(&[72, 105, u32::MAX]),
        Err(DecodeError::OutOfVocabulary {
            position: 2,
            id: u32::MAX
        })
    );

    // The first offender is reported, even when a later one exists.
    assert_eq!(
        tokenizer.decode(&[300, 400]),
        Err(DecodeError::OutOfVocabulary {
            position: 0,
            id: 300
        })
    );
}

#[test]
fn out_of_vocabulary_takes_precedence_over_invalid_bytes() {
    // 255 is a valid token that cannot start a character, and 256 is not a
    // token at all. The ID range is checked first, so the caller learns about
    // the impossible ID rather than about the UTF-8 it would have caused.
    assert_eq!(
        ByteTokenizer::new().decode(&[255, 256]),
        Err(DecodeError::OutOfVocabulary {
            position: 1,
            id: 256
        })
    );
}

#[test]
fn valid_bytes_that_are_not_utf8_report_a_utf8_error() {
    let tokenizer = ByteTokenizer::new();

    // 0xFF never appears in UTF-8, in any position.
    let error = tokenizer.decode(&[255]).expect_err("0xFF is not UTF-8");
    let DecodeError::InvalidUtf8(source) = error else {
        panic!("expected a UTF-8 error, got {error:?}");
    };
    assert_eq!(source.valid_up_to(), 0);
    assert_eq!(source.error_len(), Some(1));

    // 0xC3 announces a two-byte character whose second byte never arrives.
    // This is the shape of failure a partially generated sequence produces.
    let error = tokenizer
        .decode(&[195])
        .expect_err("truncated é is not UTF-8");
    let DecodeError::InvalidUtf8(source) = error else {
        panic!("expected a UTF-8 error, got {error:?}");
    };
    assert_eq!(source.valid_up_to(), 0);
    assert_eq!(
        source.error_len(),
        None,
        "a truncated tail is incomplete, not malformed"
    );

    // Decoding is strict: no replacement characters, no valid prefix returned.
    assert!(matches!(
        tokenizer.decode(&[72, 105, 195]),
        Err(DecodeError::InvalidUtf8(_))
    ));
}

#[test]
fn errors_describe_themselves() {
    let tokenizer = ByteTokenizer::new();

    let out_of_vocabulary = tokenizer.decode(&[0, 256]).unwrap_err();
    let message = out_of_vocabulary.to_string();
    assert!(message.contains("256"), "{message}");
    assert!(message.contains('1'), "{message}");
    assert!(out_of_vocabulary.source().is_none());

    let invalid_utf8 = tokenizer.decode(&[255]).unwrap_err();
    assert!(invalid_utf8.to_string().contains("UTF-8"));
    assert!(
        invalid_utf8.source().is_some(),
        "the underlying Utf8Error should stay reachable"
    );

    // The error type is usable in the usual boxed-error plumbing.
    let boxed: Box<dyn Error> = Box::new(invalid_utf8);
    assert!(!boxed.to_string().is_empty());
}

#[test]
fn readme_example_matches_the_api() {
    // Keep this in step with the usage snippet in README.md.
    let tokenizer = ByteTokenizer::new();

    let ids = tokenizer.encode("Hi 🦀");
    assert_eq!(ids, vec![72, 105, 32, 240, 159, 166, 128]);

    let text = tokenizer.decode(&ids).expect("valid round trip");
    assert_eq!(text, "Hi 🦀");

    match tokenizer.decode(&[255]) {
        Err(DecodeError::InvalidUtf8(error)) => assert_eq!(error.valid_up_to(), 0),
        other => panic!("expected an invalid-UTF-8 error, got {other:?}"),
    }
}
