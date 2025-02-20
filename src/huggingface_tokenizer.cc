/*!
 *  Copyright (c) 2025 by Contributors
 * \file huggingface_tokenizer.cc
 * \brief Huggingface tokenizer
 */
#include <tokenizers_rust.h>
#include <tokenizers_cpp.h>

namespace tokenizers
{

	/*!
	 * \brief A simple c++ header of tokenizer via C API.
	 */
	class RustTokenizer : public Tokenizer, public rust_impl::Tokenizer
	{
	public:
		using api = rust_impl::Tokenizer;
		inline explicit RustTokenizer(std::shared_ptr<rust_impl::SharedTokenizerHandle> handle) : rust_impl::Tokenizer(handle)
		{
#ifdef COMPILE_WASM_RUNTIME
			setenv("TOKENIZERS_PARALLELISM", "false", true);
#endif
		}

		inline ~RustTokenizer()
		{
		}

		static RustTokenizer from_file(std::string_view path)
		{
			std::shared_ptr<rust_impl::SharedTokenizerHandle> handle = std::make_shared<rust_impl::SharedTokenizerHandle>();
			handle->operator void*& () = tokenizers_new_from_file(path.data(), path.size());
			return RustTokenizer(handle);
		}

		static RustTokenizer from_json(std::string_view json)
		{
			std::shared_ptr<rust_impl::SharedTokenizerHandle> handle = std::make_shared<rust_impl::SharedTokenizerHandle>();
			handle->operator void*& () = tokenizers_new_from_str(json.data(), json.size());
			return RustTokenizer(handle);
		}

		static RustTokenizer from_byte_level_bpe(std::string_view vocab, std::string_view merges, std::string_view added_tokens)
		{
			std::shared_ptr<rust_impl::SharedTokenizerHandle> handle = std::make_shared<rust_impl::SharedTokenizerHandle>();
			handle->operator void*& () = tokenizers_new_from_byte_level_bpe(vocab.data(), vocab.size(), merges.data(), merges.size(), added_tokens.data(), added_tokens.size());
			return RustTokenizer(handle);
		}

		struct AutoPayload
		{
			std::vector<void*> payloads;

			inline AutoPayload() : payloads() {}

			AutoPayload(const AutoPayload&) = delete;

			inline AutoPayload(AutoPayload&& _Other)
			{
				payloads.swap(_Other.payloads);
			}

			inline ~AutoPayload()
			{
				auto& pool = rust::HandlePool::instance();
				for (size_t i = 0; i < payloads.size(); i++)
				{
					pool.delete_handle(payloads[i]);
				}
			}
		};

		std::vector<std::string_view> convert_string_list(std::vector<rust::String>& arr)
		{
			std::vector<std::string_view> s;
			s.reserve(arr.size());
			for (size_t i = 0; i < arr.size(); i++)
			{
				s.emplace_back(arr[i]);
			}
			return s;
		}

		// use i32 to be consistent with sentencepiece
		Encoding Encode(std::string_view text, bool add_special_tokens) final
		{
			auto encoding = api::encode(text, add_special_tokens);
			std::vector<std::string_view> tokens = convert_string_list(encoding.tokens);

			auto& pool = rust::HandlePool::instance();
			std::shared_ptr<AutoPayload> payload = std::make_shared<AutoPayload>();
			payload->payloads.push_back(pool.register_handle(encoding.get_handle()));

			Encoding result = { {{.ids = encoding.ids,
								 .type_ids = encoding.type_ids,
								 .tokens = tokens,
								 .special_tokens_mask = encoding.special_tokens_mask,
								 .attention_mask = encoding.attention_mask},
								{.payload = payload}} };

			return result;
		}

		EncodingBatch EncodeBatch(const std::vector<std::string_view>& texts, bool add_special_tokens) final
		{
			auto encodings = api::encode(texts, add_special_tokens);

			auto& pool = rust::HandlePool::instance();
			std::shared_ptr<AutoPayload> payload = std::make_shared<AutoPayload>();

			EncodingBatch result = { {}, {.payload = payload} };

			result.encodings.reserve(encodings.size());

			for (size_t i = 0; i < encodings.size(); i++)
			{
				payload->payloads.push_back(pool.register_handle(encodings[i].get_handle()));

				std::vector<std::string_view> tokens = convert_string_list(encodings[i].tokens);

				result.encodings.emplace_back(BaseEncode{ .ids = encodings[i].ids, .type_ids = encodings[i].type_ids, .tokens = tokens, .special_tokens_mask = encodings[i].special_tokens_mask, .attention_mask = encodings[i].attention_mask });
			}

			result.update();

			return result;
		}

		inline Decoding convert(rust::String&& s)
		{
			auto& pool = rust::HandlePool::instance();
			std::shared_ptr<AutoPayload> payload = std::make_shared<AutoPayload>();
			payload->payloads.push_back(pool.register_handle(s.get_handle()));

			Decoding result = { .payload = s };
			result.handle = payload;

			return result;
		}

		// use i32 to be consistent with sentencepiece
		Decoding Decode(array_view<uint32_t> ids, bool skip_special_tokens) final
		{
			return convert(api::decode(ids, skip_special_tokens));
		}

		// use i32 to be consistent with sentencepiece
		DecodingBatch DecodeBatch(const std::vector<array_view<uint32_t>>& ids_batch, bool skip_special_tokens) final
		{
			auto decoded = api::decode(ids_batch, skip_special_tokens);
			DecodingBatch result;
			result.reserve(decoded.size());

			for (size_t i = 0; i < decoded.size(); i++)
			{
				result.emplace_back(convert(std::move(decoded[i])));
			}

			return result;
		}

		size_t GetVocabSize() final
		{
			return api::get_vocab_size();
		}

		Decoding IdToToken(uint32_t id) final
		{
			auto token = api::id_to_token(id);

			auto& pool = rust::HandlePool::instance();
			std::shared_ptr<AutoPayload> payload = std::make_shared<AutoPayload>();
			payload->payloads.push_back(pool.register_handle(token.get_handle()));

			Decoding result = { .payload = token };
			result.handle = payload;

			return result;
		}

		uint32_t TokenToId(std::string_view token) final
		{
			return api::token_to_id(token);
		}

	private:
	};

	std::unique_ptr<Tokenizer> Tokenizer::FromBlobJSONFile(std::string_view json_file)
	{
		return std::make_unique<RustTokenizer>(RustTokenizer::from_file(json_file));
	}

	std::unique_ptr<Tokenizer> Tokenizer::FromBlobJSON(std::string_view json)
	{
		return std::make_unique<RustTokenizer>(RustTokenizer::from_json(json));
	}

	std::unique_ptr<Tokenizer> Tokenizer::FromBlobByteLevelBPE(std::string_view vocab,
		std::string_view merges,
		std::string_view added_tokens)
	{
		return std::make_unique<RustTokenizer>(RustTokenizer::from_byte_level_bpe(
			vocab, merges, added_tokens));
	}
} // namespace tokenizers
