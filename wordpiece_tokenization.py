from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace


# Create WordPiece tokenizer
tokenizer = Tokenizer(
    WordPiece(unk_token="[UNK]")
)

# Use whitespace pre-tokenization
tokenizer.pre_tokenizer = Whitespace()


# Create WordPiece trainer
trainer = WordPieceTrainer(
    vocab_size=100,
    min_frequency=1,
    special_tokens=[
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[PAD]",
        "[MASK]"
    ]
)


# Train tokenizer using data.txt
tokenizer.train(
    ["data.txt"],
    trainer
)


# Save vocabulary
tokenizer.model.save(".")


# Save tokenizer
tokenizer.save("tokenizer.json")


# Test input
text = "Students use technology to improve their learning."

output = tokenizer.encode(text)


# Display results
print("=" * 60)
print("              WORDPIECE TOKENIZER")
print("=" * 60)

print("\nInput Text:")
print(text)

print("\nTokens:")
print(output.tokens)

print("\nToken IDs:")
print(output.ids)

print("\nNumber of Tokens:")
print(len(output.tokens))

print("\nVocabulary Size:")
print(tokenizer.get_vocab_size())

print("=" * 60)