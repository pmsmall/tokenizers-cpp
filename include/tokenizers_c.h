/*!
 *  Copyright (c) 2025 by Contributors
 * \file tokenizers_c.h
 * \brief C binding to tokenizers rust library
 */
#ifndef TOKENIZERS_C_H_
#define TOKENIZERS_C_H_

#include <stddef.h>
#include <stdint.h>

#include <utility>

namespace rust
{
	struct Vec
	{
		void* ptr;
		size_t cap;
		size_t len;
		size_t type_size;

		inline void swap(Vec& _Other) noexcept {
			std::swap(ptr, _Other.ptr);
			std::swap(cap, _Other.cap);
			std::swap(ptr, _Other.ptr);
			std::swap(type_size, _Other.type_size);
		}
	};

	struct ArrayHandle
	{
		void* ptr;
		size_t len;

		inline void swap(ArrayHandle& _Other) noexcept {
			std::swap(ptr, _Other.ptr);
			std::swap(len, _Other.len);
		}
	};
} // namespace rust

namespace tokenizers
{
	typedef void* TokenizerHandle;
	typedef void* EncodingHandle;

	using CustomAllocator = void* (*)(size_t, void*);
	using CustomAllocatorArgs = void*;
	using CustomEmplaceBackArray = void (*)(void*, void*, size_t);
	using CustomConvertArrayHandle = ::rust::ArrayHandle(*)(void*);
	using CustomConvertArrayHandleOffset = ::rust::ArrayHandle(*)(void*, size_t offset);
} // namespace tokenizers

// The C API
#ifdef __cplusplus
extern "C"
{
#endif
	namespace tokenizers
	{
		TokenizerHandle tokenizers_new_from_str(const char* input_cstr, uintptr_t len);

		TokenizerHandle tokenizers_new_from_file(const char* path, uintptr_t len);

		TokenizerHandle tokenizers_new_from_byte_level_bpe(
			const char* input_vocab_str, uintptr_t len_vocab,
			const char* input_merges_str, uintptr_t len_merges,
			const char* input_added_tokens_str, uintptr_t len_added_tokens);

		EncodingHandle tokenizers_encode(TokenizerHandle handle, const char* input_cstr,
			uintptr_t len, int32_t add_special_tokens);

		::rust::ArrayHandle tokenizers_encoding_ids(EncodingHandle encoding_handle);

		::rust::ArrayHandle tokenizers_encoding_type_ids(EncodingHandle encoding_handle);

		::rust::ArrayHandle tokenizers_encoding_special_tokens_mask(EncodingHandle encoding_handle);

		::rust::ArrayHandle tokenizers_encoding_attention_mask(EncodingHandle encoding_handle);

		void tokenizers_encoding_tokens(
			EncodingHandle encoding_handle,
			CustomAllocator allocator,
			CustomAllocatorArgs allocator_args,
			CustomEmplaceBackArray emplace_back);

		::rust::Vec tokenizers_encode_batch(TokenizerHandle handle,
			const void* input_cstr,
			uintptr_t num_seqs,
			int32_t add_special_tokens,
			CustomConvertArrayHandleOffset convert_array_offset);

		::rust::Vec tokenizers_decode(TokenizerHandle handle, const uint32_t* input_ids,
			uintptr_t len, int32_t skip_special_tokens);

		::rust::Vec tokenizers_decode_batch(TokenizerHandle handle, const void* input_ids,
			size_t raws, int32_t skip_special_tokens,
			CustomConvertArrayHandleOffset convert_array_offset);

		size_t tokenizers_get_vocab_size(TokenizerHandle handle);

		::rust::Vec tokenizers_id_to_token(TokenizerHandle handle, uint32_t id);

		uint32_t tokenizers_token_to_id(TokenizerHandle handle, const char* token, uintptr_t len);

		void tokenizers_free(TokenizerHandle handle);
		void tokenizers_encoding_free(EncodingHandle handle);
		void tokenizers_encoding_free_with_args(const char* ptr, size_t len, size_t capacity);
		void tokenizers_encodings_free(::rust::Vec* handle);
		void tokenizers_exported_string_free(::rust::Vec* handle);
		void tokenizers_exported_strings_free(::rust::Vec* handle);
		void tokenizers_exported_strings_free_without_string_free(::rust::Vec* handle);
	} // namespace tokenizers

#ifdef __cplusplus
}

#include <string>
#include <vector>

namespace tokenizers
{
	void* alloc_string(size_t len, void* args);

	::rust::ArrayHandle fetch_string(void* arr, size_t offset);

	template <class _Ty, class _Alloc = std::allocator<_Ty>>
	void* alloc_vector(size_t len, void* args)
	{
		std::vector<_Ty, _Alloc>* arr = reinterpret_cast<std::vector<_Ty, _Alloc>*>(args);
		arr->resize(len);
		return arr->data();
	}

	template <class _Ty, class _Alloc>
	inline CustomAllocator alloc_vector_warp(const std::vector<_Ty, _Alloc>&)
	{
		return alloc_vector<_Ty, _Alloc>;
	}

	template <class _Ty, class _Alloc = std::allocator<_Ty>>
	void* reserve_vector(size_t len, void* args)
	{
		std::vector<_Ty, _Alloc>* arr = reinterpret_cast<std::vector<_Ty, _Alloc>*>(args);
		arr->reserve(len);
		return arr->data();
	}

	template <class _Ty, class _Alloc>
	inline CustomAllocator reserve_vector_warp(const std::vector<_Ty, _Alloc>&)
	{
		return reserve_vector<_Ty, _Alloc>;
	}

	template <class _Ty, class _Alloc = std::allocator<_Ty>>
	void emplace_back(void* args, void* ptr, size_t len)
	{
		std::vector<_Ty, _Alloc>* arr = reinterpret_cast<std::vector<_Ty, _Alloc>*>(args);
		arr->emplace_back(reinterpret_cast<_Ty::value_type*>(ptr), len);
	}

	template <class _Ty, class _Alloc>
	inline CustomEmplaceBackArray emplace_back_warp(const std::vector<_Ty, _Alloc>& t)
	{
		return emplace_back<_Ty, _Alloc>;
	}

	template<class _String>
	constexpr bool is_string_type_v = false;

	template <class _Elem, class _Traits, class _Alloc>
	constexpr bool is_string_type_v<std::basic_string<_Elem, _Traits, _Alloc>> = true;

	template <class _Elem, class _Traits>
	constexpr bool is_string_type_v<std::basic_string_view<_Elem, _Traits>> = true;

	template <class _Ty, class _Alloc>
	constexpr bool is_string_type_v<std::vector<_Ty, _Alloc>> = true;

	template<class _Array>
	constexpr bool is_array_type_v = false;

	template <class _Elem, class _Traits>
	constexpr bool is_array_type_v<std::basic_string_view<_Elem, _Traits>> = true;

	template <class _Ty, class _Alloc>
	constexpr bool is_array_type_v<std::vector<_Ty, _Alloc>> = true;

	template<class _String, typename std::enable_if_t<is_string_type_v<_String>, int > = 0>
	::rust::ArrayHandle get_subarray(void* args, size_t index)
	{
		std::vector<_String>* arr = reinterpret_cast<std::vector<_String>*>(args);
		auto& s = arr->at(index);
		return { const_cast<_String::value_type*>(s.data()),s.size() };
	}

	template<class _Ty>
	inline CustomConvertArrayHandleOffset get_subarray_warp(const std::vector<_Ty>&)
	{
		return get_subarray<_Ty>;
	}

	
} // namespace tokenizers

#endif
#endif // TOKENIZERS_C_H_
