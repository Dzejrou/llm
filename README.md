# tiny-llm

A small language model built from scratch in Rust, for the purpose of
understanding one rather than of using one. The goal is to be able to follow the
entire path

```text
text -> token IDs -> vectors -> attention -> logits -> loss -> gradients -> updated parameters
```

through code that is readable end to end, rather than through a framework. The
implementation grows one phase at a time; this repository currently contains the
first steps: a tokenizer, and the construction of next-token training examples
from its output.

## What is here today

A byte tokenizer: the boundary where text becomes the integer *token IDs* a
model consumes. A model never receives strings. It receives IDs, uses them to
look up rows in a parameter table, and produces a score for each ID it is
allowed to emit. The set of legal IDs is the model's *vocabulary*.

This tokenizer takes the simplest possible vocabulary: the UTF-8 encoding of the
text, one byte per token. There are 256 byte values, so the vocabulary is fixed
at 256 and, unlike a learned vocabulary such as byte-pair encoding, needs no
training pass before it can be used. Any text encodes, and any text that encoded
decodes back byte for byte.

Next to it, a data module turns those IDs into training examples. Each example pairs a
window of input tokens with the tokens that followed them, which is what a
next-token model learns to predict. Training and validation data are drawn from
separate parts of the token sequence, so no example uses tokens from both.

## Build and test

Requires a Rust toolchain supporting the 2024 edition (1.85 or newer); it is
developed against current stable.

```bash
cargo build
cargo test                                  # unit, integration and documentation tests
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

There are no dependencies, and no binary target yet — there is not yet a command
worth exposing.

## Continuous integration

Every pull request targeting `main` — and every push to `main`, including the
commit a merge produces — runs the checks above on a fresh GitHub-hosted Ubuntu
runner, as defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
The job, `Rust checks`, installs stable Rust with rustfmt and Clippy, prints the
toolchain versions, and then runs:

```bash
cargo +stable fmt --check
cargo +stable clippy --locked --all-targets -- -D warnings
cargo +stable test --locked                 # includes the documentation tests
```

`+stable` pins the toolchain rather than trusting the runner's default, and
`--locked` fails instead of quietly changing `Cargo.lock`. Any failing command
fails the job. Pushing another commit to a pull request starts a fresh run.

Results appear in the checks section at the bottom of the pull request's
*Conversation* tab and in its *Checks* tab. Each step's log is under the
repository's *Actions* tab.

What this guarantees is narrow: the proposed code is formatted, lint-free, and
passes its tests, checked the same way every time. It does not judge whether the
mathematics is right, the architecture sound, or the explanations clear. That is
still the job of code review.

## Usage

```rust
use tiny_llm::{ByteTokenizer, DecodeError};

let tokenizer = ByteTokenizer::new();

// Encoding cannot fail: a &str is valid UTF-8, and every byte is a token.
let ids = tokenizer.encode("Hi 🦀");
assert_eq!(ids, vec![72, 105, 32, 240, 159, 166, 128]);

// Decoding valid IDs recovers the text exactly.
assert_eq!(tokenizer.decode(&ids).unwrap(), "Hi 🦀");

// 0xFF is a real token but never legal UTF-8, so decoding says so rather
// than substituting a replacement character.
assert!(matches!(
    tokenizer.decode(&[255]),
    Err(DecodeError::InvalidUtf8(_)),
));
```

The same example is checked as a test, in
[`tests/tokenizer.rs`](tests/tokenizer.rs), so it cannot drift away from the
API; so is the next-token example [below](#next-token-examples). The API documentation carries further examples that `cargo test` also runs;
read them with:

```bash
cargo doc --open
```

## Why token count differs from character count

A token here is a **byte**, not a character. UTF-8 spends one byte on ASCII, two
on most accented Latin letters, three on most CJK characters, and four on emoji:

| Text    | Characters | Tokens | IDs                          |
| ------- | ---------- | ------ | ---------------------------- |
| `Hi`    | 2          | 2      | `[72, 105]`                  |
| `é`     | 1          | 2      | `[195, 169]`                 |
| `🦀`    | 1          | 4      | `[240, 159, 166, 128]`       |

This is not a rounding detail. A model sees a fixed number of token positions at
once — its context window — so the same window holds noticeably less non-ASCII
text than English. It also means a single character can be split across
positions, and the model has to learn to reassemble characters from bytes before
it can learn anything about words. A later, trained vocabulary will buy shorter
sequences at the cost of a training step; the token ID type is already wide
enough (`u32`) for such a vocabulary.

## Why decoding is strict

Encoding never fails. Decoding can fail in two genuinely different ways, and the
tokenizer reports them as two different errors:

- `DecodeError::OutOfVocabulary` — an ID above 255 names no byte at all. That is
  a bug in the caller or a mismatched vocabulary size, so the ID and its
  position in the sequence are reported. It is never narrowed by a cast, which
  would quietly keep the low eight bits and invent a byte nobody asked for.
- `DecodeError::InvalidUtf8` — every ID was a byte, but together they are not
  valid UTF-8, for example a multibyte character cut short. This one is not a
  bug: once the model is generating, it emits bytes, and nothing constrains
  those bytes to form valid text. An untrained model produces this constantly.

Nothing is replaced with `U+FFFD` and nothing is truncated to the valid prefix.
Deciding how to *display* partially valid generated output is a policy, and it
belongs to whatever eventually does the displaying; the tokenizer's job is to
report accurately what it was given.

## Next-token examples

A language model is trained to predict the next token, and the text itself
supplies the correct answers: at every position, the right prediction is the
token that actually came next. The `tiny_llm::data` module builds examples by
reading a window of `T + 1` consecutive token IDs twice. The first `T` tokens
are the *inputs*. The last `T`, shifted one position along, are the *targets*:

```text
tokens:   [10, 20, 30, 40]      context length T = 3
inputs:   [10, 20, 30]
targets:  [20, 30, 40]          targets[i] is the token that followed inputs[i]
```

`T` inputs need `T + 1` tokens, because the last target is one position past the
last input. `next_token_windows(tokens, T)` slides this window along a slice one
position at a time. A slice of `N` tokens therefore yields `max(N - T, 0)`
examples, and when `N > T` the last one starts at position `N - T - 1`. Each
example borrows its inputs and targets from the original slice, so no tokens are
copied. The iterator builds each example only when it is requested.

Part of the data is held out of training so that its loss can later show how the
model does on text it was not fitted to. The order of operations matters:

1. **Split the token sequence** with `split_tokens(tokens, train_len)`. Training
   gets exactly `tokens[..train_len]`, and validation gets `tokens[train_len..]`.
   The split point is a token index, not a ratio, so the boundary is exact and
   reproducible.
2. **Then build windows from each portion separately.** Neighbouring windows
   overlap, so dividing an already-built list of windows would put the same
   source positions on both sides. Because each portion is windowed on its own,
   no example crosses the boundary.

```rust
use tiny_llm::ByteTokenizer;
use tiny_llm::data::{DataError, next_token_windows, split_tokens};

let ids = ByteTokenizer::new().encode("abcdefgh");

// 1. Split first: five tokens for training, three held out.
let split = split_tokens(&ids, 5).unwrap();
assert_eq!(split.train, [97, 98, 99, 100, 101]); // "abcde"
assert_eq!(split.validation, [102, 103, 104]); // "fgh"

// 2. Then window each portion on its own, here with T = 2.
let mut train = next_token_windows(split.train, 2).unwrap();
assert_eq!(train.len(), 3); // N - T = 5 - 2
let first = train.next().unwrap();
assert_eq!(first.inputs(), [97, 98]); // "ab"
assert_eq!(first.targets(), [98, 99]); // "bc"

let mut validation = next_token_windows(split.validation, 2).unwrap();
let only = validation.next().unwrap();
assert_eq!(only.inputs(), [102, 103]); // "fg"
assert_eq!(only.targets(), [103, 104]); // "gh"
assert!(validation.next().is_none());

// Too few tokens for the context is not an error: there are no examples.
assert_eq!(next_token_windows(split.validation, 3).unwrap().len(), 0);

// Splitting at either end is valid and leaves one portion empty.
let all_train = split_tokens(&ids, ids.len()).unwrap();
assert!(all_train.validation.is_empty());
assert_eq!(next_token_windows(all_train.validation, 2).unwrap().len(), 0);

// Arguments that can never describe valid data are errors.
assert_eq!(
    split_tokens(&ids, 9),
    Err(DataError::SplitOutOfBounds { train_len: 9, len: 8 }),
);
assert_eq!(
    next_token_windows(&ids, 0).unwrap_err(),
    DataError::ZeroContextLength,
);
```

The edge cases follow from that. A split at `0` or at `tokens.len()` is valid,
including for empty input; it just leaves one portion empty. A portion with `T`
or fewer tokens (an empty one included) yields no examples rather than an error,
and nothing is ever padded. Only a split point past the end and a context length
of zero are rejected. Whether a portion holds *enough* data to train or evaluate
on is for the future training command to decide.

This layer treats token IDs as plain integers. It does not know the vocabulary
size, and it never decodes anything. A split or a window edge can fall inside a
multibyte character, leaving a slice that is not valid UTF-8 on its own. That
does not matter here, because the model consumes IDs, not text. A contiguous
split keeps token *positions* apart. It cannot stop a passage that repeats in
the source text from appearing on both sides.

## Scope

Deliberately absent: normalization, special tokens, padding, vocabulary
training, and any abstraction over tokenizers; on the data side, file loading,
shuffling, random sampling, and batching. Models, training, and generation
arrive in later phases.
