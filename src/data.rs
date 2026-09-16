//! Next-token examples: turning a stream of token IDs into training data.
//!
//! # Where the targets come from
//!
//! Supervised training needs examples of the form "given this input, the right
//! answer is that". A language model is trained to predict the next token, and
//! for that task nobody has to write the answers down: the text supplies them.
//! At every position, the correct prediction is simply the token that actually
//! came next. A stretch of token IDs becomes an example by reading it twice,
//! once as the *inputs* and once, shifted one position along, as the *targets*:
//!
//! ```text
//! source:   [10, 20, 30, 40]        context length T = 3
//! inputs:   [10, 20, 30]
//! targets:      [20, 30, 40]
//! ```
//!
//! Each column is one question: after `10` comes `20`, after `20` comes `30`,
//! after `30` comes `40`. In general `targets[i]` is the source token that
//! follows `inputs[i]`. The shift by one is the whole point. Without it every
//! target would equal its input, and a model could score perfectly by copying.
//!
//! One example therefore holds `T` predictions. A model that predicts
//! `targets[i]` may use `inputs[..=i]`, the tokens up to and including
//! position `i`, but nothing after it; otherwise it could read the answer off
//! `inputs[i + 1]`. Enforcing that restriction is the model's job. This
//! module only builds the examples.
//!
//! # Why `T` inputs need `T + 1` tokens
//!
//! For a window starting at source position `s`, the inputs are positions
//! `s..s+T` and the targets are positions `s+1..s+T+1`. The last target lies
//! one past the last input, so the two together span `T + 1` positions,
//! `s..s+T+1`. Such a window fits in a sequence of `N` tokens only if
//! `s + T + 1 <= N`, that is, `s <= N - T - 1`.
//!
//! When `N > T`, the valid starts are therefore `0, 1, ..., N - T - 1`, which
//! is `N - T` windows. The last one starts at `N - T - 1`, and its final target
//! is the final token, at position `N - 1`. A window starting one position
//! later would need a target at position `N`, and there is none. When
//! `N <= T` no start is valid and there are no examples; nothing is ever padded
//! to make one. The two cases together give `max(N - T, 0)` examples. The code
//! computes that as `N.saturating_sub(T)` and never forms `T + 1` or an
//! unchecked `N - T`, so it cannot overflow, even for `T == usize::MAX`.
//!
//! # Examples are borrowed views, not copies
//!
//! Neighbouring windows overlap almost completely: the windows starting at `s`
//! and `s + 1` share `T` of their `T + 1` tokens, and within one window the
//! inputs and targets share `T - 1`. Copying every example out would store
//! each token roughly `2T` times. Instead, an [`Example`] borrows one slice of
//! `T + 1` consecutive source tokens, and [`Example::inputs`] and
//! [`Example::targets`] are two sub-slices of it that differ only in where they
//! start and end. A slice is a pointer and a length, so an example has the same
//! small, fixed size whatever `T` is, and no token is ever copied.
//! [`NextTokenWindows`] creates the examples one at a time, as they are
//! requested, instead of collecting all `N - T` of them into a list. The borrow
//! checker guarantees that the source tokens outlive every example taken from
//! them.
//!
//! # Split first, then window
//!
//! Validation data is kept out of parameter updates so that the loss on it can
//! later say something about text the model was not fitted to. That only
//! works if no validation position also appears in a training example. Windows
//! overlap, so cutting an already-built list of windows cannot guarantee that:
//! two neighbouring windows that land on opposite sides of the cut still share
//! `T` source positions.
//!
//! [`split_tokens`] therefore cuts the *token stream* into two disjoint,
//! contiguous portions, and windows are then built separately from each:
//!
//! ```text
//! tokens:      [0, 1, 2, 3 | 4, 5, 6, 7]         split at 4, T = 2
//! train:       [0, 1] -> [1, 2]    [1, 2] -> [2, 3]
//! validation:  [4, 5] -> [5, 6]    [5, 6] -> [6, 7]
//! ```
//!
//! A window iterator only reads the slice it is given, so no example crosses
//! the cut. The transition `3 -> 4` belongs to neither portion. The two windows
//! that would have straddled the cut, starting at positions 2 and 3, are the
//! price of a clean boundary. Windowing the unsplit stream would produce them,
//! which is why the split must come first.
//!
//! The split point is an explicit token index, not a fraction, so the boundary
//! is exact and reproducible. Turning a ratio such as "90% for training" into
//! an index is a policy for the caller. A contiguous split keeps *positions*
//! apart; it cannot promise that the text never repeats itself, and a passage
//! that occurs on both sides of the cut will appear in both portions.
//!
//! # Token IDs are opaque here
//!
//! This module never asks what an ID means. It does not know the vocabulary
//! size, does not reject IDs above 255, and decodes nothing. A split point or a
//! window edge may fall in the middle of a multibyte UTF-8 character, leaving a
//! slice that is not valid text on its own. That is fine. The model consumes
//! IDs, not text, and the IDs are unchanged from the sequence they were cut
//! from. Decoding matters only when a person wants to read something, and that
//! is the tokenizer's job (see
//! [`ByteTokenizer::decode`](crate::ByteTokenizer::decode)).
//!
//! # Example
//!
//! Encode, split at an explicit index, then window each portion separately:
//!
//! ```
//! use tiny_llm::ByteTokenizer;
//! use tiny_llm::data::{next_token_windows, split_tokens};
//!
//! let ids = ByteTokenizer::new().encode("abcdefgh");
//!
//! // 1. Split the token stream: five tokens for training, three held out.
//! let split = split_tokens(&ids, 5)?;
//! assert_eq!(split.train, [97, 98, 99, 100, 101]); // "abcde"
//! assert_eq!(split.validation, [102, 103, 104]); // "fgh"
//!
//! // 2. Window each portion on its own, here with T = 2.
//! let mut train = next_token_windows(split.train, 2)?;
//! assert_eq!(train.len(), 3); // N - T = 5 - 2
//! let first = train.next().unwrap();
//! assert_eq!(first.inputs(), [97, 98]); // "ab"
//! assert_eq!(first.targets(), [98, 99]); // "bc"
//!
//! let mut validation = next_token_windows(split.validation, 2)?;
//! let only = validation.next().unwrap();
//! assert_eq!(only.inputs(), [102, 103]); // "fg"
//! assert_eq!(only.targets(), [103, 104]); // "gh"
//! assert!(validation.next().is_none());
//!
//! // A portion too short for the context is not an error; it has no examples.
//! assert_eq!(next_token_windows(split.validation, 3)?.len(), 0);
//!
//! // Splitting at either end is valid and leaves one portion empty.
//! let all_train = split_tokens(&ids, ids.len())?;
//! assert!(all_train.validation.is_empty());
//! assert_eq!(next_token_windows(all_train.validation, 2)?.len(), 0);
//! # Ok::<(), tiny_llm::data::DataError>(())
//! ```

use std::error::Error;
use std::fmt;
use std::iter::FusedIterator;
use std::ops::Range;

use crate::TokenId;

/// A problem with the arguments passed to [`split_tokens`] or
/// [`next_token_windows`].
///
/// Only arguments that can never describe valid data are errors. A split that
/// leaves one portion empty, or a sequence too short for its context, is valid
/// input with an unremarkable result, so neither is reported here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataError {
    /// [`split_tokens`] was asked for more training tokens than the sequence
    /// contains.
    ///
    /// The split point is not clamped to the end of the sequence, because
    /// silently producing a smaller training portion than requested would hide
    /// the caller's mistake.
    SplitOutOfBounds {
        /// The requested number of training tokens.
        train_len: usize,
        /// The length of the sequence that was to be split.
        len: usize,
    },
    /// [`next_token_windows`] was asked for a context length of zero.
    ///
    /// A zero-length example has no inputs and no targets, so it asks nothing
    /// to learn from. A zero here almost certainly means a missing or
    /// mistyped setting, so it is reported instead of producing empty examples.
    ZeroContextLength,
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SplitOutOfBounds { train_len, len } => write!(
                f,
                "cannot take {train_len} training tokens from a sequence of {len} \
                 (the split point must be at most {len})"
            ),
            Self::ZeroContextLength => {
                f.write_str("context length must be at least 1 token, but was 0")
            }
        }
    }
}

impl Error for DataError {}

/// A token sequence cut into a training portion followed by a validation
/// portion.
///
/// Both fields borrow from the sequence passed to [`split_tokens`], and nothing
/// is copied. `train` followed by `validation` is exactly the original
/// sequence, and either field may be empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Split<'a> {
    /// `tokens[..train_len]`: the tokens training examples are built from.
    pub train: &'a [TokenId],
    /// `tokens[train_len..]`: the held-out tokens, used only to evaluate.
    pub validation: &'a [TokenId],
}

/// Cuts `tokens` into a training portion of exactly `train_len` tokens and a
/// validation portion of the rest.
///
/// Call this *before* building windows, then call [`next_token_windows`] once
/// for each portion; the [module documentation](self#split-first-then-window)
/// explains why the order matters.
///
/// Both endpoints are valid: `train_len == 0` puts everything in validation,
/// and `train_len == tokens.len()` puts everything in training. Empty input
/// can therefore be split at index 0. Whether a portion is large enough to
/// train or evaluate on is for the caller to decide.
///
/// # Errors
///
/// [`DataError::SplitOutOfBounds`] if `train_len > tokens.len()`.
///
/// ```
/// use tiny_llm::data::{DataError, split_tokens};
///
/// let tokens = [10, 20, 30, 40];
///
/// let split = split_tokens(&tokens, 3)?;
/// assert_eq!(split.train, [10, 20, 30]);
/// assert_eq!(split.validation, [40]);
///
/// assert_eq!(
///     split_tokens(&tokens, 5),
///     Err(DataError::SplitOutOfBounds { train_len: 5, len: 4 })
/// );
/// # Ok::<(), DataError>(())
/// ```
pub fn split_tokens(tokens: &[TokenId], train_len: usize) -> Result<Split<'_>, DataError> {
    // `split_at_checked` returns `None` exactly when `train_len > tokens.len()`,
    // so the bounds check and the cut itself cannot disagree.
    let (train, validation) =
        tokens
            .split_at_checked(train_len)
            .ok_or(DataError::SplitOutOfBounds {
                train_len,
                len: tokens.len(),
            })?;
    Ok(Split { train, validation })
}

/// One next-token example: `T` input tokens and, for each of them, the token
/// that followed it.
///
/// An example holds a single borrowed window of `T + 1` consecutive source
/// tokens. [`inputs`](Self::inputs) and [`targets`](Self::targets) are two
/// overlapping views of that window; see the [module
/// documentation](self#examples-are-borrowed-views-not-copies).
///
/// Examples are only created by [`NextTokenWindows`], so every example meets
/// two conditions: its inputs and targets have the same non-zero length `T`,
/// and `targets()[i]` is the source token immediately after `inputs()[i]`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Example<'a> {
    /// `T + 1` consecutive source tokens, with `T >= 1`.
    window: &'a [TokenId],
}

impl<'a> Example<'a> {
    /// The `T` tokens the model reads: source positions `s..s+T`.
    ///
    /// The returned slice borrows from the original token sequence, not from
    /// this example, so it may outlive the example.
    #[must_use]
    pub fn inputs(&self) -> &'a [TokenId] {
        // Every token of the window except the last.
        &self.window[..self.window.len() - 1]
    }

    /// The `T` tokens the model should predict: source positions
    /// `s+1..s+T+1`.
    ///
    /// `targets()[i]` is the answer for the prediction made at `inputs()[i]`.
    /// Like [`inputs`](Self::inputs), the slice borrows from the original
    /// sequence.
    #[must_use]
    pub fn targets(&self) -> &'a [TokenId] {
        // Every token of the window except the first.
        &self.window[1..]
    }
}

impl fmt::Debug for Example<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Show the two views the API exposes, not the internal window.
        f.debug_struct("Example")
            .field("inputs", &self.inputs())
            .field("targets", &self.targets())
            .finish()
    }
}

/// Iterator over every next-token [`Example`] in a slice, created by
/// [`next_token_windows`].
///
/// Starting positions are visited in ascending order with a stride of one, so
/// the sequence of examples is fully determined by the slice and the context
/// length. There is no shuffling, sampling, padding, or wraparound. The
/// iterator knows exactly how many examples remain, and returns `None` forever
/// once it is exhausted. Cloning it copies only its position, not the tokens.
#[derive(Debug, Clone)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct NextTokenWindows<'a> {
    tokens: &'a [TokenId],
    context_len: usize,
    /// Starting positions not yet visited; `0..max(N - T, 0)` when created.
    starts: Range<usize>,
}

impl<'a> Iterator for NextTokenWindows<'a> {
    type Item = Example<'a>;

    fn next(&mut self) -> Option<Example<'a>> {
        let start = self.starts.next()?;
        // `start <= N - T - 1`, so `start + T <= N - 1`: the addition cannot
        // overflow and the inclusive range ends on the last target in bounds.
        let window = &self.tokens[start..=start + self.context_len];
        Some(Example { window })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.starts.size_hint()
    }
}

impl ExactSizeIterator for NextTokenWindows<'_> {}

impl FusedIterator for NextTokenWindows<'_> {}

/// Returns an iterator over every next-token example of context length
/// `context_len` (`T`) in `tokens`.
///
/// The example starting at position `s` has inputs `tokens[s..s+T]` and
/// targets `tokens[s+1..s+T+1]`. Every valid start is visited once, in
/// ascending order, for a total of `max(tokens.len() - T, 0)` examples. The
/// [module documentation](self#why-t-inputs-need-t--1-tokens) explains the
/// count.
///
/// If `tokens` has `T` or fewer tokens, which includes empty input and any
/// oversized `T` up to `usize::MAX`, the iterator is simply empty. Only
/// `tokens` is read. To keep training and validation examples apart, call
/// this once for each portion returned by [`split_tokens`], never on the
/// unsplit sequence.
///
/// Nothing is computed until the iterator is advanced, and each example
/// borrows from `tokens`, so the call itself takes constant time and memory.
///
/// # Errors
///
/// [`DataError::ZeroContextLength`] if `context_len` is 0.
///
/// ```
/// use tiny_llm::data::next_token_windows;
///
/// let tokens = [10, 20, 30, 40];
/// let mut windows = next_token_windows(&tokens, 3)?;
/// assert_eq!(windows.len(), 1);
///
/// let example = windows.next().unwrap();
/// assert_eq!(example.inputs(), [10, 20, 30]);
/// assert_eq!(example.targets(), [20, 30, 40]);
/// assert!(windows.next().is_none());
/// # Ok::<(), tiny_llm::data::DataError>(())
/// ```
pub fn next_token_windows(
    tokens: &[TokenId],
    context_len: usize,
) -> Result<NextTokenWindows<'_>, DataError> {
    if context_len == 0 {
        return Err(DataError::ZeroContextLength);
    }

    // There are `max(N - T, 0)` valid starts. `saturating_sub` computes that
    // without the underflow of `N - T` when `N < T`, and without the overflow
    // of `T + 1` when `T == usize::MAX`.
    let window_count = tokens.len().saturating_sub(context_len);
    Ok(NextTokenWindows {
        tokens,
        context_len,
        starts: 0..window_count,
    })
}
