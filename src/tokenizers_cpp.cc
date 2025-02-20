/*!
 *  Copyright (c) 2025 by Contributors
 * \file tokenizers_cpp.cc
 */
#include "tokenizers_cpp.h"

namespace tokenizers {
	torch::Device global::CUDA0(torch::DeviceType::CUDA);
	torch::Device global::CPU(torch::DeviceType::CPU);
}  // namespace tokenizers

template<class _T>
void copy(_T& src, _T& dst)
{
	dst = src;
}

tokenizers::EncodingBatch tokenizers::Tokenizer::EncodeBatch(const std::vector<std::string_view>& texts, bool add_special_tokens)
{
	EncodingBatch res;

	res.encodings.reserve(texts.size());

	do
	{
		auto first = Encode(texts[0], add_special_tokens);

		copy<tokenizers::options>(first, res);

		res.encodings.emplace_back(std::move(first));
	} while (false);


	for (size_t i = 1; i < texts.size(); ++i)
	{
		res.encodings.emplace_back(Encode(texts[i], add_special_tokens));
	}

	return res;
}

inline std::vector<std::vector<uint32_t>> u32tensorTo2DVec(const torch::Tensor& ids) {
	std::vector<std::vector<uint32_t>> tensor;
	auto&& sizes = ids.sizes();
	tensor.reserve(sizes[0]);
	for (size_t i = 0; i < sizes[0]; ++i)
	{
		auto&& sub = ids[i];
		auto dataPtr = reinterpret_cast<uint32_t*>(sub.mutable_data_ptr());
		tensor.emplace_back(dataPtr, dataPtr + sub.size(0));
	}
	return tensor;
}

inline std::vector<std::vector<uint32_t>> tensorTo2DVec(const torch::Tensor& ids) {
	if (ids.dtype() != torch::kUInt32)
	{
		if (!ids.device().is_cpu())
			return u32tensorTo2DVec(ids.to(torch::kUInt32).to(tokenizers::global::CPU));
		else
			return u32tensorTo2DVec(ids.to(torch::kUInt32));
	}
	else
	{
		if (!ids.device().is_cpu())
			return u32tensorTo2DVec(ids.to(tokenizers::global::CPU));
		else
			return u32tensorTo2DVec(ids);
	}
}

inline std::vector<tokenizers::array_view<uint32_t>> vecToView(const std::vector<std::vector<uint32_t>>& vec)
{
	std::vector<tokenizers::array_view<uint32_t>> res;
	res.reserve(vec.size());
	for (size_t i = 0; i < vec.size(); ++i)
	{
		auto& sub = vec[0];
		res.emplace_back(sub.data(), sub.size());
	}
	return res;
}

inline std::vector<tokenizers::array_view<uint32_t>> u32tensorToView(const torch::Tensor& ids) {
	std::vector<tokenizers::array_view<uint32_t>> tensor;
	auto&& sizes = ids.sizes();
	tensor.reserve(sizes[0]);
	for (size_t i = 0; i < sizes[0]; ++i)
	{
		auto&& sub = ids[i];
		auto dataPtr = reinterpret_cast<uint32_t*>(sub.mutable_data_ptr());
		tensor.emplace_back(dataPtr, dataPtr + sub.size(0));
	}
	return tensor;
}

tokenizers::Decoding tokenizers::Tokenizer::Decode(const torch::Tensor& ids, bool skip_special_token)
{
	if (ids.dtype() == torch::kUInt32 && ids.is_cpu())
		return Decode(u32tensorToView(ids)[0], skip_special_token);
	else
		return Decode(tensorTo2DVec(ids)[0], skip_special_token);
}

tokenizers::Decoding tokenizers::Tokenizer::Decode(const std::vector<uint32_t>& ids, bool skip_special_token)
{
	return Decode({ ids.data(),ids.size() }, skip_special_token);
}

tokenizers::DecodingBatch tokenizers::Tokenizer::DecodeBatch(const torch::Tensor& ids_batch, bool skip_special_token)
{
	if (ids_batch.dtype() == torch::kUInt32 && ids_batch.is_cpu())
		return DecodeBatch(u32tensorToView(ids_batch), skip_special_token);
	else
		return DecodeBatch(tensorTo2DVec(ids_batch), skip_special_token);
}

tokenizers::DecodingBatch tokenizers::Tokenizer::DecodeBatch(
	const std::vector<tokenizers::array_view<uint32_t>>& ids_batch, bool skip_special_token)
{
	tokenizers::DecodingBatch res;
	res.reserve(ids_batch.size());

	for (size_t i = 0; i < ids_batch.size(); ++i)
	{
		res.emplace_back(Decode(ids_batch[i], skip_special_token));
	}

	return res;
}

tokenizers::DecodingBatch tokenizers::Tokenizer::DecodeBatch(
	const std::vector<std::vector<uint32_t>>& ids_batch, bool skip_special_token) {
	return DecodeBatch(vecToView(ids_batch), skip_special_token);
}
