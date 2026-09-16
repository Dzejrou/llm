//! Behavioural tests for the public next-token data API.
//!
//! Like the tokenizer tests, these use the crate only through its public
//! interface. They pin down the one-position shift between inputs and targets,
//! the number and order of examples, the inputs that produce no examples, and
//! the guarantee that no example crosses the train/validation split.

use std::error::Error;
use std::ptr;

use tiny_llm::data::{DataError, NextTokenWindows, next_token_windows, split_tokens};
use tiny_llm::{ByteTokenizer, DecodeError, TokenId};

/// An example's inputs and targets, copied out so they compare against
/// literals.
type OwnedExample = (Vec<TokenId>, Vec<TokenId>);

/// Creates the window iterator for a context length the test knows is valid.
fn windows(tokens: &[TokenId], context_len: usize) -> NextTokenWindows<'_> {
    next_token_windows(tokens, context_len).expect("context length is positive")
}

/// Collects every example of `tokens` at `context_len`, in iteration order.
fn examples(tokens: &[TokenId], context_len: usize) -> Vec<OwnedExample> {
    windows(tokens, context_len)
        .map(|example| (example.inputs().to_vec(), example.targets().to_vec()))
        .collect()
}

/// Every `(input, target)` pair an example asks the model to learn, across
/// all examples of `tokens`.
fn transitions(tokens: &[TokenId], context_len: usize) -> Vec<(TokenId, TokenId)> {
    windows(tokens, context_len)
        .flat_map(|example| {
            example
                .inputs()
                .iter()
                .copied()
                .zip(example.targets().iter().copied())
        })
        .collect()
}

#[test]
fn targets_are_the_inputs_shifted_by_one() {
    // The example from the issue: three predictions from four tokens.
    assert_eq!(
        examples(&[10, 20, 30, 40], 3),
        vec![(vec![10, 20, 30], vec![20, 30, 40])]
    );
}

#[test]
fn every_valid_start_is_visited_in_ascending_order() {
    let tokens = [11, 12, 13, 14, 15, 16];

    // N = 6, T = 2: starts 0 through N - T - 1 = 3, so four examples.
    assert_eq!(
        examples(&tokens, 2),
        vec![
            (vec![11, 12], vec![12, 13]),
            (vec![12, 13], vec![13, 14]),
            (vec![13, 14], vec![14, 15]),
            // The last window ends exactly on the last token.
            (vec![14, 15], vec![15, 16]),
        ]
    );

    // The same, stated as the general rule for each start `s`.
    for (start, example) in windows(&tokens, 2).enumerate() {
        assert_eq!(example.inputs(), &tokens[start..start + 2]);
        assert_eq!(example.targets(), &tokens[start + 1..start + 3]);
    }
}

#[test]
fn context_length_one_yields_the_adjacent_token_pairs() {
    // These single-position examples are what a bigram model learns from.
    let tokens = [5, 9, 5, 7, 2];

    let pairs: Vec<(TokenId, TokenId)> = windows(&tokens, 1)
        .map(|example| {
            assert_eq!(example.inputs().len(), 1);
            assert_eq!(example.targets().len(), 1);
            (example.inputs()[0], example.targets()[0])
        })
        .collect();

    assert_eq!(pairs, vec![(5, 9), (9, 5), (5, 7), (7, 2)]);
    let adjacent: Vec<(TokenId, TokenId)> = tokens.windows(2).map(|w| (w[0], w[1])).collect();
    assert_eq!(pairs, adjacent);
}

#[test]
fn a_sequence_needs_more_than_t_tokens_to_yield_an_example() {
    // N == T + 1: exactly one example.
    assert_eq!(examples(&[1, 2, 3, 4], 3).len(), 1);
    // N == T: the inputs would fit, but the last target would not.
    assert!(examples(&[1, 2, 3], 3).is_empty());
    // N < T.
    assert!(examples(&[1, 2], 3).is_empty());
    // Empty input.
    assert!(examples(&[], 1).is_empty());
    assert!(examples(&[], 3).is_empty());
}

#[test]
fn example_count_is_n_minus_t_or_zero() {
    let tokens: Vec<TokenId> = (100..108).collect();

    for len in 0..=tokens.len() {
        let slice = &tokens[..len];
        for context_len in 1..=len + 2 {
            let expected = len.saturating_sub(context_len);
            let iter = windows(slice, context_len);
            assert_eq!(iter.len(), expected, "N = {len}, T = {context_len}");

            let produced: Vec<_> = iter.collect();
            assert_eq!(produced.len(), expected, "N = {len}, T = {context_len}");
            for example in produced {
                assert_eq!(example.inputs().len(), context_len);
                assert_eq!(example.targets().len(), context_len);
            }
        }
    }
}

#[test]
fn remaining_length_is_exact_and_exhaustion_is_permanent() {
    let tokens = [1, 2, 3, 4, 5];
    let mut iter = windows(&tokens, 2);

    for remaining in (1..=3).rev() {
        assert_eq!(iter.len(), remaining);
        assert_eq!(iter.size_hint(), (remaining, Some(remaining)));
        assert!(iter.next().is_some());
    }

    assert_eq!(iter.len(), 0);
    assert!(iter.next().is_none());
    assert!(iter.next().is_none());
    assert_eq!(iter.len(), 0);
}

#[test]
fn zero_context_length_is_rejected() {
    assert_eq!(
        next_token_windows(&[1, 2, 3], 0).unwrap_err(),
        DataError::ZeroContextLength
    );
    // Rejected even when there are no tokens to window at all.
    assert_eq!(
        next_token_windows(&[], 0).unwrap_err(),
        DataError::ZeroContextLength
    );
}

#[test]
fn oversized_context_lengths_yield_nothing_without_overflowing() {
    // `T + 1` would overflow for `usize::MAX`, and `N - T` would underflow for
    // all of these. Neither may happen; the result is simply no examples.
    for context_len in [usize::MAX, usize::MAX - 1, 1_000_000] {
        let mut iter = windows(&[1, 2, 3], context_len);
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());

        assert!(windows(&[], context_len).next().is_none());
    }
}

#[test]
fn split_at_every_valid_index_partitions_the_sequence() {
    let tokens: Vec<TokenId> = (0..6).map(|i| i * 10).collect();

    for train_len in 0..=tokens.len() {
        let split = split_tokens(&tokens, train_len).expect("index is in bounds");
        assert_eq!(split.train, &tokens[..train_len]);
        assert_eq!(split.validation, &tokens[train_len..]);
        assert_eq!([split.train, split.validation].concat(), tokens);
    }
}

#[test]
fn split_accepts_both_endpoints_and_empty_input() {
    let tokens = [7, 8, 9];

    let all_validation = split_tokens(&tokens, 0).unwrap();
    assert!(all_validation.train.is_empty());
    assert_eq!(all_validation.validation, [7, 8, 9]);

    let all_train = split_tokens(&tokens, tokens.len()).unwrap();
    assert_eq!(all_train.train, [7, 8, 9]);
    assert!(all_train.validation.is_empty());

    let empty = split_tokens(&[], 0).unwrap();
    assert!(empty.train.is_empty());
    assert!(empty.validation.is_empty());

    // An empty portion is valid data; it just has no examples.
    assert_eq!(windows(all_validation.train, 1).len(), 0);
    assert_eq!(windows(all_train.validation, 1).len(), 0);
}

#[test]
fn split_rejects_indices_past_the_end() {
    let tokens = [7, 8, 9];

    assert_eq!(
        split_tokens(&tokens, 4),
        Err(DataError::SplitOutOfBounds {
            train_len: 4,
            len: 3
        })
    );
    assert_eq!(
        split_tokens(&tokens, usize::MAX),
        Err(DataError::SplitOutOfBounds {
            train_len: usize::MAX,
            len: 3
        })
    );
    assert_eq!(
        split_tokens(&[], 1),
        Err(DataError::SplitOutOfBounds {
            train_len: 1,
            len: 0
        })
    );
}

#[test]
fn splits_and_examples_borrow_the_source_tokens() {
    let tokens: Vec<TokenId> = (0..8).collect();

    // `ptr::eq` on slices compares both the address and the length, so these
    // hold only if the results are views of `tokens` itself, not copies.
    let split = split_tokens(&tokens, 3).unwrap();
    assert!(ptr::eq(split.train, &tokens[..3]));
    assert!(ptr::eq(split.validation, &tokens[3..]));

    for (start, example) in windows(&tokens, 3).enumerate() {
        assert!(ptr::eq(example.inputs(), &tokens[start..start + 3]));
        assert!(ptr::eq(example.targets(), &tokens[start + 1..start + 4]));
    }

    // The slices borrow from `tokens`, not from the example that produced them.
    let inputs = windows(&tokens, 2).next().unwrap().inputs();
    assert_eq!(inputs, [0, 1]);
}

#[test]
fn train_and_validation_examples_do_not_cross_the_split() {
    let tokens = [0, 1, 2, 3, 4, 5, 6, 7];
    let split = split_tokens(&tokens, 4).unwrap();

    let train = examples(split.train, 2);
    let validation = examples(split.validation, 2);

    assert_eq!(
        train,
        vec![(vec![0, 1], vec![1, 2]), (vec![1, 2], vec![2, 3])]
    );
    assert_eq!(
        validation,
        vec![(vec![4, 5], vec![5, 6]), (vec![5, 6], vec![6, 7])]
    );

    // Training examples read only positions 0..=3, validation only 4..=7.
    // (The tokens equal their positions here.)
    for (inputs, targets) in &train {
        assert!(inputs.iter().chain(targets).all(|&id| id <= 3));
    }
    for (inputs, targets) in &validation {
        assert!(inputs.iter().chain(targets).all(|&id| id >= 4));
    }

    // The transition across the cut is in neither portion.
    assert!(!transitions(split.train, 2).contains(&(3, 4)));
    assert!(!transitions(split.validation, 2).contains(&(3, 4)));

    // Windowing before splitting would have leaked it, which is why the split
    // comes first.
    assert!(transitions(&tokens, 2).contains(&(3, 4)));
}

#[test]
fn iteration_is_deterministic() {
    let tokens: Vec<TokenId> = (0..20).map(|i| (i * 7) % 11).collect();

    assert_eq!(examples(&tokens, 4), examples(&tokens, 4));

    // A clone continues from the same position and yields the same examples.
    let mut original = windows(&tokens, 4);
    assert!(original.nth(5).is_some());
    let copy = original.clone();
    assert!(original.eq(copy));
}

#[test]
fn ids_outside_the_byte_range_are_ordinary_data() {
    let tokens = [256, 0, u32::MAX, 70_000, u32::MAX];

    assert_eq!(
        split_tokens(&tokens, 2).unwrap().validation,
        [u32::MAX, 70_000, u32::MAX]
    );
    assert_eq!(
        examples(&tokens, 2),
        vec![
            (vec![256, 0], vec![0, u32::MAX]),
            (vec![0, u32::MAX], vec![u32::MAX, 70_000]),
            (vec![u32::MAX, 70_000], vec![70_000, u32::MAX]),
        ]
    );
}

#[test]
fn cuts_inside_a_multibyte_character_are_allowed() {
    let tokenizer = ByteTokenizer::new();
    // 'é' is two bytes and the crab is four.
    let ids = tokenizer.encode("é🦀");
    assert_eq!(ids, vec![195, 169, 240, 159, 166, 128]);

    // Split between the two bytes of 'é'.
    let split = split_tokens(&ids, 1).unwrap();
    assert_eq!(split.train, [195]);
    assert_eq!(split.validation, [169, 240, 159, 166, 128]);

    // Neither portion is text on its own. That matters to the tokenizer,
    // which refuses to decode them, but not to the data layer.
    assert!(matches!(
        tokenizer.decode(split.validation),
        Err(DecodeError::InvalidUtf8(_))
    ));

    // Windows cut through the crab's four bytes all the same.
    assert_eq!(
        examples(split.validation, 2),
        vec![
            (vec![169, 240], vec![240, 159]),
            (vec![240, 159], vec![159, 166]),
            (vec![159, 166], vec![166, 128]),
        ]
    );
    // The training portion is valid but too short to yield an example.
    assert!(examples(split.train, 1).is_empty());
}

#[test]
fn errors_describe_themselves() {
    let out_of_bounds = split_tokens(&[1, 2, 3], 5).unwrap_err();
    let message = out_of_bounds.to_string();
    assert!(message.contains('5'), "{message}");
    assert!(message.contains('3'), "{message}");
    assert!(out_of_bounds.source().is_none());

    let zero = next_token_windows(&[1, 2, 3], 0).unwrap_err();
    let message = zero.to_string();
    assert!(message.contains("context length"), "{message}");
    assert!(zero.source().is_none());
    assert_eq!(format!("{zero:?}"), "ZeroContextLength");

    // The error type works with the usual boxed-error plumbing.
    let boxed: Box<dyn Error> = Box::new(out_of_bounds);
    assert!(!boxed.to_string().is_empty());
}

#[test]
fn examples_debug_as_their_inputs_and_targets() {
    let example = windows(&[10, 20, 30], 2).next().unwrap();
    assert_eq!(
        format!("{example:?}"),
        "Example { inputs: [10, 20], targets: [20, 30] }"
    );
}

#[test]
fn readme_data_example_matches_the_api() {
    // Keep this in step with the next-token example in README.md.
    let ids = ByteTokenizer::new().encode("abcdefgh");

    let split = split_tokens(&ids, 5).expect("5 <= 8");
    assert_eq!(split.train, [97, 98, 99, 100, 101]);
    assert_eq!(split.validation, [102, 103, 104]);

    let train = examples(split.train, 2);
    assert_eq!(train.len(), 3);
    assert_eq!(train[0], (vec![97, 98], vec![98, 99]));

    assert_eq!(
        examples(split.validation, 2),
        vec![(vec![102, 103], vec![103, 104])]
    );
    assert!(examples(split.validation, 3).is_empty());

    let all_train = split_tokens(&ids, ids.len()).expect("the end is a valid split");
    assert!(all_train.validation.is_empty());
    assert!(examples(all_train.validation, 2).is_empty());

    assert_eq!(
        split_tokens(&ids, 9),
        Err(DataError::SplitOutOfBounds {
            train_len: 9,
            len: 8
        })
    );
    assert_eq!(
        next_token_windows(&ids, 0).unwrap_err(),
        DataError::ZeroContextLength
    );
}
