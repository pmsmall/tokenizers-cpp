/*!
 *  Copyright (c) 2025 by Contributors
 * \file sentencepiece_tokenizer.cc
 * \brief Sentencepice tokenizer
 */
#include <sentencepiece_processor.h>
#include <tokenizers_cpp.h>

#include <cassert>

namespace tokenizers
{

#ifdef MLC_ENABLE_SENTENCEPIECE_TOKENIZER

	class SentencePieceTokenizer : public Tokenizer
	{
	public:
		explicit SentencePieceTokenizer(std::string_view model_blob)
		{
			sentence_piece_.LoadFromSerializedProto({ model_blob.data(), model_blob.size() });
		}

		Encoding Encode(std::string_view text, bool add_special_tokens) final
		{
			std::shared_ptr<std::vector<int32_t>> tokens = std::make_shared<std::vector<int32_t>>();
			sentence_piece_.Encode({ text.data(), text.size() }, tokens.get()).IgnoreError();
			return { {{.ids = array_view<uint32_t>{reinterpret_cast<uint32_t*>(tokens->data()), tokens->size()}}, {.payload = tokens}} };
		}

		Decoding Decode(array_view<uint32_t> ids, bool skip_special_tokens) final
		{
			std::string text;
			sentence_piece_.Decode(array_view<int32_t>(reinterpret_cast<const int32_t*>(ids.data()), ids.size()), &text).IgnoreError();

			Decoding result = { {.buff = std::move(text)} };
			result.payload = result.buff.value();

			return result;
		}

		Decoding Decode(const std::vector<uint32_t>& ids, bool skip_special_tokens) final
		{
			std::string text;
			sentence_piece_.Decode(array_view<int32_t>(reinterpret_cast<const int32_t*>(ids.data()), ids.size()), &text).IgnoreError();

			Decoding result = { {.buff = std::move(text)} };
			result.payload = result.buff.value();

			return result;
		}

		size_t GetVocabSize() final
		{
			auto size = sentence_piece_.GetPieceSize();
			assert(size > 0);
			return size;
		}

		Decoding IdToToken(uint32_t id) final { return { .payload = sentence_piece_.IdToPiece(id) }; }

		uint32_t TokenToId(std::string_view token) final { return sentence_piece_.PieceToId({ token.data(), token.size() }); }

	private:
		// the tokenizer
		sentencepiece::SentencePieceProcessor sentence_piece_;
	};

	std::unique_ptr<Tokenizer> Tokenizer::FromBlobSentencePiece(std::string_view model_blob)
	{
		return std::make_unique<SentencePieceTokenizer>(model_blob);
	}
#else
	std::unique_ptr<Tokenizer> Tokenizer::FromBlobSentencePiece(const std::string& model_blob)
	{
		assert(false);
		throw;
	}
#endif // MLC_ENABLE_SENTENCEPIECE_TOKENIZER

} // namespace tokenizers
