# tiny-llm

A small language model built from scratch in Rust, for the purpose of
understanding one rather than of using one. The goal is to be able to follow the
entire path

```text
text -> token IDs -> vectors -> attention -> logits -> loss -> gradients -> updated parameters
```

through code that is readable end to end, rather than through a framework. The
implementation grows one phase at a time; this repository currently contains the
very first step, the tokenizer.

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
API. The API documentation carries further examples that `cargo test` also runs;
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

## Scope

Deliberately absent: normalization, special tokens, padding, vocabulary
training, and any abstraction over tokenizers. Models, training, and generation
arrive in later phases.
