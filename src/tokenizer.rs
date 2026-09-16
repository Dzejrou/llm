//! Byte tokenization: the boundary between human-readable text and model input.
//!
//! # Why a tokenizer exists
//!
//! A language model never sees a `String`. It sees a sequence of integers, the
//! *token IDs*, which it uses to index a parameter table: one learned row of
//! numbers per possible token. A tokenizer is therefore nothing more than an
//! agreed-upon mapping between text and a finite set of IDs, plus the promise
//! that the mapping can be undone.
//!
//! The set of IDs the model is allowed to see is its *vocabulary*, and the size
//! of that vocabulary is a real architectural constraint: it fixes the number
//! of rows in the model's embedding table and the number of scores the model
//! produces for every prediction. Choosing a vocabulary is the first modelling
//! decision, so it is worth making the simplest possible choice first.
//!
//! # Bytes as tokens
//!
//! [`ByteTokenizer`] uses the UTF-8 encoding of the text and treats each byte
//! as one token. Because a byte has exactly 256 possible values, the vocabulary
//! is fixed at [`ByteTokenizer::VOCAB_SIZE`] and requires no training step: the
//! mapping is known before a single character of a corpus is read.
//!
//! The price is that **a token is a byte, not a character**. ASCII characters
//! occupy one byte each, but `'é'` occupies two and most emoji occupy four, so
//! a short string can produce many more tokens than it has characters. That
//! matters downstream, because the model's context window is counted in tokens:
//! non-ASCII text consumes more of it. Later phases may replace this tokenizer
//! with a trained vocabulary (for example byte-pair encoding) whose tokens span
//! several bytes; [`TokenId`] is deliberately wider than a byte so that such a
//! vocabulary needs no change to the types flowing through the model.
//!
//! # Why decoding can fail
//!
//! Encoding cannot fail: a `&str` is valid UTF-8 by construction, and every
//! byte of it is a legal token. Decoding is the interesting direction, because
//! the IDs handed to it need not come from [`ByteTokenizer::encode`] at all —
//! once the model is generating, they are whatever the model sampled. Two
//! independent things can then go wrong, and this module reports them as two
//! distinct variants of [`DecodeError`]:
//!
//! 1. An ID may not name a byte at all (anything above 255). This is a bug in
//!    the caller or a model configured with the wrong vocabulary size.
//! 2. The IDs may all be valid bytes, yet the byte sequence they form may not
//!    be valid UTF-8 — a half-finished multibyte character, for instance. A
//!    freshly initialized model samples bytes essentially at random, so this is
//!    an expected, routine outcome rather than a bug.
//!
//! Decoding here is strict: it never substitutes replacement characters and
//! never truncates to the valid prefix. Hiding case 2 inside the tokenizer
//! would take away information that a later text-display layer needs in order
//! to choose its own policy.

use std::error::Error;
use std::fmt;
use std::str::Utf8Error;

/// A single token identifier, as consumed by the model.
///
/// Byte tokens only ever occupy `0..=255`, so the extra width is not needed to
/// store them. It is here to keep two ideas apart: a `u8` is raw *storage*,
/// while a `TokenId` is a *position in a vocabulary*. The two happen to
/// coincide for byte tokenization and will stop coinciding as soon as a larger
/// vocabulary arrives, at which point only the tokenizer needs to change.
pub type TokenId = u32;

/// Something that went wrong while turning token IDs back into text.
///
/// The two variants are kept separate on purpose; see the [module
/// documentation](self#why-decoding-can-fail) for why the distinction matters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// A token ID lies outside the byte vocabulary, i.e. it is greater than
    /// 255 and therefore names no byte.
    ///
    /// The offending value and its index in the input slice are reported so
    /// that the caller can point at the exact token rather than at the whole
    /// sequence.
    OutOfVocabulary {
        /// Index of the offending ID within the slice passed to
        /// [`ByteTokenizer::decode`].
        position: usize,
        /// The ID that is not a member of the vocabulary.
        id: TokenId,
    },
    /// Every ID named a byte, but those bytes are not a valid UTF-8 string.
    ///
    /// The wrapped [`Utf8Error`] carries where the problem starts
    /// ([`Utf8Error::valid_up_to`]) and whether the sequence was merely
    /// truncated ([`Utf8Error::error_len`] returning `None`), which is what
    /// distinguishes "stopped mid-character" from "genuinely malformed".
    InvalidUtf8(Utf8Error),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfVocabulary { position, id } => write!(
                f,
                "token id {id} at position {position} is outside the byte vocabulary \
                 (valid ids are 0..={max})",
                max = ByteTokenizer::VOCAB_SIZE - 1
            ),
            Self::InvalidUtf8(source) => {
                write!(f, "decoded bytes are not valid UTF-8: {source}")
            }
        }
    }
}

impl Error for DecodeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OutOfVocabulary { .. } => None,
            Self::InvalidUtf8(source) => Some(source),
        }
    }
}

/// A tokenizer whose vocabulary is the 256 possible byte values.
///
/// The type holds no data: the mapping is defined by UTF-8 itself, so there is
/// nothing to train and nothing to configure. It is a type rather than a pair
/// of free functions so that later tokenizers, which *will* carry a learned
/// vocabulary, can present the same shape of API.
///
/// # Examples
///
/// ```
/// use tiny_llm::ByteTokenizer;
///
/// let tokenizer = ByteTokenizer::new();
///
/// // ASCII: one byte, and therefore one token, per character.
/// assert_eq!(tokenizer.encode("Hi"), vec![72, 105]);
///
/// // Non-ASCII: two characters, but six tokens, because UTF-8 spends two
/// // bytes on 'é' and four on the emoji.
/// let ids = tokenizer.encode("é🦀");
/// assert_eq!(ids.len(), 6);
/// assert_eq!("é🦀".chars().count(), 2);
///
/// // Decoding is the exact inverse for anything encode produced.
/// assert_eq!(tokenizer.decode(&ids)?, "é🦀");
/// # Ok::<(), tiny_llm::DecodeError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    /// The number of distinct token IDs this tokenizer can ever produce.
    ///
    /// Every byte value is a token and no other token exists, so this is
    /// exactly the number of byte values. A model built on this tokenizer
    /// embeds `VOCAB_SIZE` rows and scores `VOCAB_SIZE` candidates per step.
    pub const VOCAB_SIZE: usize = 256;

    /// Creates a tokenizer.
    ///
    /// There is no state to initialize; the constructor exists for symmetry
    /// with tokenizers that will need one.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Returns [`Self::VOCAB_SIZE`], for callers holding a tokenizer value
    /// rather than naming the type.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        Self::VOCAB_SIZE
    }

    /// Encodes text as the sequence of its UTF-8 byte values, in order.
    ///
    /// This cannot fail. A `&str` is guaranteed to be valid UTF-8, and every
    /// one of its bytes is a member of the vocabulary by definition. Empty
    /// input yields an empty sequence.
    ///
    /// The returned length is the *byte* length of the text, which is at least
    /// its character count and may be up to four times larger.
    ///
    /// ```
    /// # use tiny_llm::ByteTokenizer;
    /// // 'é' is U+00E9, encoded by UTF-8 as the two bytes 0xC3 0xA9.
    /// assert_eq!(ByteTokenizer::new().encode("é"), vec![195, 169]);
    /// ```
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        // Widening u8 -> u32 is lossless, so `from` is enough and no cast is
        // involved; this direction of the conversion can never misbehave.
        text.as_bytes().iter().copied().map(TokenId::from).collect()
    }

    /// Decodes token IDs back into text, or explains why they are not text.
    ///
    /// The IDs are validated before anything is built from them: an ID above
    /// 255 is rejected with [`DecodeError::OutOfVocabulary`] rather than being
    /// narrowed with a cast, which would silently keep the low eight bits and
    /// invent a byte the caller never asked for. Only once every ID is known to
    /// be a byte are the bytes checked, as a whole, for UTF-8 validity.
    ///
    /// Both checks are necessary and neither implies the other: `[300]` is out
    /// of vocabulary, while `[255]` is a perfectly good token that happens to
    /// be illegal as the first byte of a UTF-8 character.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::OutOfVocabulary`] if some ID exceeds 255. The first
    ///   such ID and its position are reported.
    /// - [`DecodeError::InvalidUtf8`] if the resulting bytes are not valid
    ///   UTF-8, including the case of a multibyte character cut short.
    ///
    /// ```
    /// # use tiny_llm::{ByteTokenizer, DecodeError};
    /// let tokenizer = ByteTokenizer::new();
    ///
    /// assert_eq!(tokenizer.decode(&[72, 105]).unwrap(), "Hi");
    ///
    /// // A byte that is never legal in UTF-8.
    /// assert!(matches!(
    ///     tokenizer.decode(&[255]),
    ///     Err(DecodeError::InvalidUtf8(_))
    /// ));
    ///
    /// // Not a byte at all.
    /// assert!(matches!(
    ///     tokenizer.decode(&[72, 256]),
    ///     Err(DecodeError::OutOfVocabulary { position: 1, id: 256 })
    /// ));
    /// ```
    pub fn decode(&self, ids: &[TokenId]) -> Result<String, DecodeError> {
        let mut bytes = Vec::with_capacity(ids.len());
        for (position, &id) in ids.iter().enumerate() {
            let byte =
                u8::try_from(id).map_err(|_| DecodeError::OutOfVocabulary { position, id })?;
            bytes.push(byte);
        }

        // UTF-8 validity is a property of the sequence, not of any single byte,
        // so it can only be decided once all of the bytes are known.
        String::from_utf8(bytes).map_err(|error| DecodeError::InvalidUtf8(error.utf8_error()))
    }
}
