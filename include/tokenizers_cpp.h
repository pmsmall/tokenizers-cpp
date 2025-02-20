/*!
 *  Copyright (c) 2025 by Contributors
 * \file tokenizers_cpp.h
 * \brief A C++ binding to common set of tokenizers
 */
#ifndef TOKENIZERS_CPP_H_
#define TOKENIZERS_CPP_H_

#include <torch/script.h>
#include <memory>
#include <string>
#include <vector>
#include <optional>

namespace tokenizers
{

	template <class _Ty>
	using array_view = std::basic_string_view<_Ty>;

	class SimpleEncodeBatchResult;

	class global
	{
	public:
		static torch::Device CUDA0;
		static torch::Device CPU;
	private:

	};

	struct BaseEncode
	{
		std::optional<array_view<uint32_t>> ids = std::nullopt;
		std::optional<array_view<uint32_t>> type_ids = std::nullopt;
		std::optional<std::vector<std::string_view>> tokens = std::nullopt;
		std::optional<array_view<uint32_t>> special_tokens_mask = std::nullopt;
		std::optional<array_view<uint32_t>> attention_mask = std::nullopt;
	};

	struct BaseEncodePayload
	{
		std::optional<std::vector<uint32_t>> ids = std::nullopt;
		std::optional<std::vector<uint32_t>> type_ids = std::nullopt;
		std::optional<std::vector<std::string_view>> tokens = std::nullopt;
		std::optional<std::vector<uint32_t>> special_tokens_mask = std::nullopt;
		std::optional<std::vector<uint32_t>> attention_mask = std::nullopt;
	};

	struct EncodeAdvancedPayload
	{
		std::shared_ptr<void> payload = NULL;
	};

	struct options
	{
		torch::Dtype type = torch::kUInt32;
		torch::Device device = global::CUDA0;
	};

	struct EncodeAdvanced : public BaseEncode, public EncodeAdvancedPayload
	{
	};

	struct Encoding :public EncodeAdvanced, public options
	{
		inline void insert(torch::jit::Kwargs &args,
						   std::string_view key,
						   array_view<uint32_t> &arr,
						   c10::TensorOptions &options)
		{
			auto tensor = torch::from_blob(const_cast<uint32_t*>(arr.data()),
				{ 1, static_cast<long long>(arr.size()) },
				options);
			if (type != torch::kUInt32)
				tensor = tensor.to(type);

			if (device != global::CPU)
				tensor = tensor.to(device);

			args.insert({ std::string(key), tensor });
		}

		inline operator torch::jit::Kwargs()
		{
			torch::jit::Kwargs args;

			c10::TensorOptions options(torch::kUInt32);

			if (ids.has_value())
			{
				insert(args, "input_ids", ids.value(), options);
			}

			if (attention_mask.has_value())
			{
				insert(args, "attention_mask", attention_mask.value(), options);
			}

			if (type_ids.has_value())
			{
				insert(args, "token_type_ids", type_ids.value(), options);
			}

			return args;
		}
	};

	struct EncodingBatch : public BaseEncodePayload, public EncodeAdvancedPayload, public options
	{
		std::vector<EncodeAdvanced> encodings;

		size_t max_len = 0;

		inline void update(std::vector<array_view<uint32_t>>& src, std::optional<std::vector<uint32_t>>& target)
		{
			size_t total = src.size() * max_len;
			std::vector<uint32_t> arr(total, 0);
			for (size_t i = 0; i < src.size(); i++)
			{
				std::memcpy(arr.data() + i * max_len, src[i].data(), src[i].size());
			}
			target = arr;
		}

		inline void update_max_len(const std::optional<array_view<uint32_t>>& arr)
		{
			auto& temp = arr.value();
			if (temp.size() > max_len)
				max_len = temp.size();
		}

		inline void update()
		{
			std::vector<array_view<uint32_t>> temp_ids, temp_attention_mask, temp_type_ids;
			temp_ids.resize(encodings.size());
			temp_attention_mask.resize(encodings.size());
			temp_type_ids.resize(encodings.size());

			bool has_ids = false, has_mask = false, has_tids = false;

			max_len = 0;

			for (size_t i = 0; i < encodings.size(); i++)
			{
				auto& e = encodings[i];
				if (e.ids.has_value())
				{
					has_ids = true;
					update_max_len(e.ids);
					temp_ids[i] = e.ids.value();
				}
				if (e.attention_mask.has_value())
				{
					has_mask = true;
					update_max_len(e.attention_mask);
					temp_attention_mask[i] = e.attention_mask.value();
				}
				if (e.type_ids.has_value())
				{
					has_tids = true;
					update_max_len(e.type_ids);
					temp_type_ids[i] = e.type_ids.value();
				}
			}

			if (has_ids)
				update(temp_ids, ids);
			if(has_mask)
				update(temp_attention_mask, attention_mask);
			if (has_tids)
				update(temp_type_ids, type_ids);
		}

		inline void updateOnce()
		{
			if (encodings.size() && !max_len)
				update();
		}

		inline void insert(torch::jit::Kwargs &args,
						   std::string_view key,
						   std::vector<uint32_t> &arr,
						   c10::TensorOptions &options)
		{

			auto tensor = torch::from_blob(const_cast<uint32_t*>(arr.data()),
				{ static_cast<long long>(arr.size() / max_len), static_cast<long long>(max_len) },
				options);
			if (type != torch::kUInt32)
				tensor = tensor.to(type);

			if (device != global::CPU)
				tensor = tensor.to(device);

			args.insert({ std::string(key), tensor });
		}

		inline operator torch::jit::Kwargs()
		{
			torch::jit::Kwargs args;

			if (encodings.size())
			{
				c10::TensorOptions options(torch::kUInt32);

				updateOnce();

				if (ids.has_value())
				{
					insert(args, "input_ids", ids.value(), options);
				}

				if (attention_mask.has_value())
				{
					insert(args, "attention_mask", attention_mask.value(), options);
				}

				if (type_ids.has_value())
				{
					insert(args, "token_type_ids", type_ids.value(), options);
				}
			}

			return args;
		}
	};

	struct DecodePayload
	{
		std::optional<std::string> buff = std::nullopt;
		std::shared_ptr<void> handle;
	};

	struct Decoding : public DecodePayload
	{
		std::string_view payload;

		inline operator std::string_view() const { return payload; }
	};

	using DecodingBatch = std::vector<Decoding>;

	inline std::vector<std::string_view> convert(DecodingBatch& decodings)
	{
		std::vector<std::string_view> res;
		res.reserve(decodings.size());
		for (size_t i = 0; i < decodings.size(); i++) res.emplace_back(decodings[i]);
	}

	/*!
	 * \brief a universal tokenizer that loads
	 *  either HF's tokenizer or sentence piece,
	 *  depending on the constructor
	 */
	class Tokenizer
	{
	public:
		/*! \brief virtual destructor */
		virtual ~Tokenizer() {}

		/*!
		 * \brief EncodeResult text into ids.
		 * \param text The input text.
		 * \returns The encoded token ids.
		 */
		virtual Encoding Encode(std::string_view text,
			bool add_special_tokens = true) = 0;

		/*!
		 * \brief EncodeResult a batch of texts into ids.
		 * \param texts The input texts.
		 * \returns The encoded token ids.
		 */
		virtual EncodingBatch EncodeBatch(const std::vector<std::string_view>& texts,
			bool add_special_tokens = true);

		/*!
		 * \brief Decode token ids into text.
		 * \param text The token ids.
		 * \returns The decoded text.
		 */
		virtual Decoding Decode(const torch::Tensor& ids, bool skip_special_token = true);

		/*!
		 * \brief Decode token ids into text.
		 * \param text The token ids.
		 * \returns The decoded text.
		 */
		virtual Decoding Decode(const std::vector<uint32_t>& ids, bool skip_special_token = true);

		/*!
		 * \brief Decode token ids into text.
		 * \param text The token ids.
		 * \returns The decoded text.
		 */
		virtual Decoding Decode(array_view<uint32_t> ids, bool skip_special_token = true) = 0;

		/*!
		 * \brief Decode a batch of token ids into texts.
		 * \param text The token ids.
		 * \returns The decoded text.
		 */
		virtual DecodingBatch DecodeBatch(
			const torch::Tensor& ids_batch, bool skip_special_token = true);

		/*!
		 * \brief Decode a batch of token ids into texts.
		 * \param text The token ids.
		 * \returns The decoded text.
		 */
		virtual DecodingBatch DecodeBatch(
			const std::vector<array_view<uint32_t>>& ids_batch, bool skip_special_token = true);

		/*!
		 * \brief Decode a batch of token ids into texts.
		 * \param text The token ids.
		 * \returns The decoded text.
		 */
		virtual DecodingBatch DecodeBatch(
			const std::vector<std::vector<uint32_t>>& ids_batch, bool skip_special_token = true);

		/*!
		 * \brief Returns the vocabulary size. Special tokens are considered.
		 */
		virtual size_t GetVocabSize() = 0;

		/*!
		 * \brief Convert the given id to its corresponding token if it exists. If
		 * not, return an empty string.
		 */
		virtual Decoding IdToToken(uint32_t token_id) = 0;

		/*!
		 * \brief Convert the given token to its corresponding id if it exists. If
		 * not, return -1.
		 */
		virtual uint32_t TokenToId(std::string_view token) = 0;

		virtual void clearCache() {}

		//---------------------------------------------------
		// Factory functions from byte-blobs
		// These factory function takes in in-memory blobs
		// so the library can be independent from filesystem
		//---------------------------------------------------
		/*!
		 * \brief Create HF tokenizer from a single in-memory json blob.
		 *
		 * \param json_blob The json blob.
		 * \return The created tokenzier.
		 */
		static std::unique_ptr<Tokenizer> FromBlobJSONFile(std::string_view json_file);

		//---------------------------------------------------
		// Factory functions from byte-blobs
		// These factory function takes in in-memory blobs
		// so the library can be independent from filesystem
		//---------------------------------------------------
		/*!
		 * \brief Create HF tokenizer from a single in-memory json blob.
		 *
		 * \param json_blob The json blob.
		 * \return The created tokenzier.
		 */
		static std::unique_ptr<Tokenizer> FromBlobJSON(std::string_view json_blob);
		/*!
		 * \brief Create BPE tokenizer
		 *
		 * \param vocab_blob The blob that contains vocabs.
		 * \param merges_blob The blob that contains the merges.
		 * \param added_tokens The added tokens.
		 * \return The created tokenizer.
		 */
		static std::unique_ptr<Tokenizer> FromBlobByteLevelBPE(std::string_view vocab_blob,
			std::string_view merges_blob,
			std::string_view added_tokens = "");
		/*!
		 * \brief Create SentencePiece.
		 *
		 * \param model_blob The blob that contains vocabs.
		 * \return The created tokenizer.
		 */
		static std::unique_ptr<Tokenizer> FromBlobSentencePiece(std::string_view model_blob);
		/*!
		 * \brief Create RWKVWorldTokenizer.
		 *
		 * \param model_blob The blob that contains vocabs.
		 * \return The created tokenizer.
		 */
		static std::unique_ptr<Tokenizer> FromBlobRWKVWorld(std::string_view model_blob);
	};
} // namespace tokenizers
#endif // TOKENIZERS_CPP_H_
