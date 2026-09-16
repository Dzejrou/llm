//! `tiny-llm`: an educational, from-scratch language model in Rust.
//!
//! The project exists to make one path fully visible:
//!
//! ```text
//! text -> token IDs -> vectors -> attention -> logits -> loss -> gradients -> updated parameters
//! ```
//!
//! This crate currently implements the first arrow only. [`tokenizer`] turns
//! text into the integer token IDs a model consumes, and turns them back again,
//! refusing to guess when the IDs do not describe text. Everything after that —
//! embeddings, the model itself, training and generation — arrives in later
//! phases and will live in its own modules; the tokenizer deliberately knows
//! nothing about them.
//!
//! # Example
//!
//! ```
//! use tiny_llm::ByteTokenizer;
//!
//! let tokenizer = ByteTokenizer::new();
//! let ids = tokenizer.encode("Hi");
//!
//! assert_eq!(ids, vec![72, 105]);
//! assert_eq!(tokenizer.decode(&ids)?, "Hi");
//! # Ok::<(), tiny_llm::DecodeError>(())
//! ```

pub mod tokenizer;

pub use tokenizer::{ByteTokenizer, DecodeError, TokenId};

/// Compiles and runs the usage example in `README.md` as a documentation test,
/// so that the README cannot drift away from the API it documents.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
