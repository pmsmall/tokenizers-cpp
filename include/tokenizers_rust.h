/*!
 *  Copyright (c) 2025 by Contributors
 * \file tokenizers_rust.h
 * \brief C++ binding to tokenizers rust library
 */
#ifndef TOKENIZERS_RUST_H_
#define TOKENIZERS_RUST_H_

#include "tokenizers_c.h"
#include "memory"
#include "exception"

namespace interface
{
	class BaseSharedHandle
	{
	public:
		virtual void* handle() = 0;
	};
}

namespace rust
{
	class SharedStringHandle : public Vec, public interface::BaseSharedHandle
	{
	public:
		inline SharedStringHandle() : Vec{ 0, 0, 0, 0 } {}

		inline SharedStringHandle(const Vec& v) : Vec(v) {}

		SharedStringHandle(const SharedStringHandle&) = delete;

		inline SharedStringHandle(SharedStringHandle&& _Other) noexcept { swap(_Other); }

		inline ~SharedStringHandle()
		{
			if (ptr)
				tokenizers::tokenizers_exported_string_free(this);
		}

		void* handle() override;

	private:
	};

	class String : public std::string_view
	{
	public:
		inline constexpr String() : std::string_view(), handle(NULL) {}

		inline String(std::shared_ptr<SharedStringHandle> handle) : std::string_view(reinterpret_cast<char*>(handle->ptr), handle->len), handle(handle)
		{
			if (!handle->ptr || sizeof(char) != handle->type_size)
				throw std::exception("handle is not vec<char>!");
		}

		inline constexpr String(const void* ptr, const size_t len) : std::string_view(reinterpret_cast<const char*>(ptr), len), handle(NULL) {}

		String(const String& _ref) : std::string_view(_ref), handle(_ref.handle) {}

		inline String(String&& _ref) noexcept
		{
			swap(_ref);
		}

		inline String& operator=(const rust::String& _Other)
		{
			std::string_view::operator=(_Other);
			handle = _Other.handle;
			return *this;
		}

		inline operator std::string_view& () { return *this; }

		inline void swap(String& _Other) noexcept
		{
			std::string_view::swap(_Other);
			handle.swap(_Other.handle);
		}

		std::shared_ptr<SharedStringHandle> get_handle() const { return handle; }

	private:
		std::shared_ptr<SharedStringHandle> handle;
	};
} // namespace rust

namespace std
{
	_EXPORT_STD inline void swap(rust::String& _Left, rust::String& _Right) noexcept
	{
		_Left.swap(_Right);
	}
} // namespace std

namespace tokenizers
{
	template <>
	constexpr bool is_string_type_v<::rust::String> = true;
} // namespace tokenizers

namespace tokenizers
{
	namespace rust_impl
	{
		template <class _Ty>
		using array_view = std::basic_string_view<_Ty>;

		enum InitType
		{
			HANDLE,
			PARENT,
			NONE
		};

		class SharedEncodingHandle : public interface::BaseSharedHandle
		{
		public:
			inline SharedEncodingHandle() : m_handle(NULL), initType(NONE) {}
			inline SharedEncodingHandle(EncodingHandle handle, InitType initType) : m_handle(handle), initType(initType) {}

			SharedEncodingHandle(const SharedEncodingHandle&) = delete;

			inline SharedEncodingHandle(SharedEncodingHandle&& _Other) noexcept
			{
				std::swap(m_handle, _Other.m_handle);
				std::swap(initType, _Other.initType);
			}

			inline ~SharedEncodingHandle()
			{
				switch (initType)
				{
				case HANDLE:
					tokenizers_encoding_free(m_handle);
					break;
				case PARENT:
					break;
				default:
					break;
				}
			}

			InitType type() const { return initType; }

			operator EncodingHandle() { return m_handle; }

			void* handle() override;

		private:
			EncodingHandle m_handle;
			InitType initType;
		};

		class SharedEncodingArrayHandle : public interface::BaseSharedHandle
		{
		public:
			inline SharedEncodingArrayHandle(::rust::Vec handle) : m_handle(handle) {}

			SharedEncodingArrayHandle(const SharedEncodingArrayHandle&) = delete;

			inline SharedEncodingArrayHandle(SharedEncodingArrayHandle&& _Other) noexcept
			{
				std::swap(m_handle, _Other.m_handle);
			}

			inline ~SharedEncodingArrayHandle()
			{
				if (m_handle.ptr)
					tokenizers_encodings_free(&m_handle);
			}

			inline operator rust::Vec& () { return m_handle; }

			void* handle() override;

		private:
			::rust::Vec m_handle;
		};

		class Encoding
		{
		public:
			inline Encoding() : handle(NULL), parent(NULL) {}

			inline Encoding(std::shared_ptr<SharedEncodingHandle> handle) : Encoding(handle, NULL)
			{
			}

			inline Encoding(
				std::shared_ptr<SharedEncodingHandle> handle,
				std::shared_ptr<SharedEncodingArrayHandle> parent) : ids(convertArray<uint32_t>(tokenizers_encoding_ids(*handle))),
				type_ids(convertArray<uint32_t>(tokenizers_encoding_type_ids(*handle))),
				tokens(),
				special_tokens_mask(convertArray<uint32_t>(tokenizers_encoding_special_tokens_mask(*handle))),
				attention_mask(convertArray<uint32_t>(tokenizers_encoding_attention_mask(*handle))),
				handle(handle),
				parent(parent)
			{
				switch (handle->type())
				{
				case PARENT:
					if (!parent)
						throw std::exception("The value of parent cannot be NULL!");
					break;
				case HANDLE:
					break;
				default:
					break;
				}

				tokenizers::tokenizers_encoding_tokens(
					*handle,
					tokenizers::reserve_vector_warp(tokens),
					&tokens,
					tokenizers::emplace_back_warp(tokens));
			}

			inline ~Encoding()
			{
			}

			inline void swap(Encoding& _Other) noexcept
			{
				ids.swap(_Other.ids);
				type_ids.swap(_Other.type_ids);
				tokens.swap(_Other.tokens);
				special_tokens_mask.swap(_Other.special_tokens_mask);
				attention_mask.swap(_Other.attention_mask);
				std::swap(handle, _Other.handle);
				parent.swap(_Other.parent);
			}

			array_view<uint32_t> ids;
			array_view<uint32_t> type_ids;
			std::vector<rust::String> tokens;
			array_view<uint32_t> special_tokens_mask;
			array_view<uint32_t> attention_mask;

			std::shared_ptr<SharedEncodingHandle> get_handle() const { return handle; }

		protected:
			template <class _Ty>
			inline array_view<_Ty> convertArray(::rust::ArrayHandle arrayHandle)
			{
				return array_view<_Ty>(reinterpret_cast<_Ty*>(arrayHandle.ptr), arrayHandle.len);
			}

		private:
			std::shared_ptr<SharedEncodingHandle> handle;

			std::shared_ptr<SharedEncodingArrayHandle> parent;
		};

		namespace Encodings
		{
			inline std::vector<Encoding> fetch(std::shared_ptr<SharedEncodingArrayHandle> handle)
			{
				std::vector<Encoding> encodings;
				encodings.reserve(handle->operator rust::Vec & ().len);

				rust::Vec& v = *handle;

				for (size_t i = 0; i < v.len; i++)
				{
					void* ptr = static_cast<char*>(v.ptr) + i * v.type_size;
					std::shared_ptr<SharedEncodingHandle> encoding = std::make_shared<SharedEncodingHandle>(ptr, PARENT);
					encodings.emplace_back(encoding, handle);
				}

				return encodings;
			}
		}

		class SharedTokenizerHandle : public interface::BaseSharedHandle
		{
		public:
			inline SharedTokenizerHandle() : m_handle(0) {}

			inline SharedTokenizerHandle(TokenizerHandle handle) : m_handle(handle) {}

			SharedTokenizerHandle(const SharedTokenizerHandle&) = delete;

			inline SharedTokenizerHandle(SharedTokenizerHandle&& _Other) noexcept
			{
				std::swap(m_handle, _Other.m_handle);
			}

			inline ~SharedTokenizerHandle()
			{
				if (m_handle)
					tokenizers_free(m_handle);
			}

			inline operator TokenizerHandle& () { return m_handle; }

			void* handle() override;

		private:
			TokenizerHandle m_handle;
		};

		class Tokenizer
		{
		public:
			inline Tokenizer(std::shared_ptr<SharedTokenizerHandle> handle) : handle(handle) {}

			static Tokenizer from_file(std::string_view path)
			{
				std::shared_ptr<SharedTokenizerHandle> handle = std::make_shared<SharedTokenizerHandle>();
				handle->operator void*& () = tokenizers_new_from_file(path.data(), path.size());
				return Tokenizer(handle);
			}

			static Tokenizer from_json(std::string_view json)
			{
				std::shared_ptr<SharedTokenizerHandle> handle = std::make_shared<SharedTokenizerHandle>();
				handle->operator void*& () = tokenizers_new_from_str(json.data(), json.size());
				return Tokenizer(handle);
			}

			static Tokenizer from_byte_level_bpe(std::string_view vocab, std::string_view merges, std::string_view added_tokens)
			{
				std::shared_ptr<SharedTokenizerHandle> handle = std::make_shared<SharedTokenizerHandle>();
				handle->operator void*& () = tokenizers_new_from_byte_level_bpe(vocab.data(), vocab.size(), merges.data(), merges.size(), added_tokens.data(), added_tokens.size());
				return Tokenizer(handle);
			}

			inline Encoding encode(std::string_view input, bool add_special_tokens = true)
			{
				auto raw_handle = tokenizers_encode(*handle, input.data(), input.size(), add_special_tokens);
				std::shared_ptr<SharedEncodingHandle> encoding_handle = std::make_shared<SharedEncodingHandle>(raw_handle, HANDLE);
				return Encoding(encoding_handle);
			}

			template <class _String, typename std::enable_if_t<is_string_type_v<_String>, int> = 0>
			inline std::vector<Encoding> encode(const std::vector<_String>& input, bool add_special_tokens = true)
			{
				return Encodings::fetch(
					std::make_shared<SharedEncodingArrayHandle>(
						tokenizers_encode_batch(
							*handle,
							&input,
							input.size(),
							add_special_tokens,
							get_subarray_warp(input))));
			}

			template <class _Array,
					  typename std::enable_if_t<is_array_type_of_v<_Array, uint32_t> || is_array_type_of_v<_Array, int32_t>, int> = 0>
			inline ::rust::String decode(const _Array & ids, bool skip_special_tokens = true)
			{
				return ::rust::String(std::make_shared<::rust::SharedStringHandle>(tokenizers_decode(*handle, ids.data(), ids.size(), skip_special_tokens)));
			}

			template <class _Array,
					  typename std::enable_if_t<is_array_type_of_v<_Array, uint32_t> || is_array_type_of_v<_Array, int32_t>, int> = 0>
			inline std::vector<::rust::String> decode(std::vector<_Array> ids, bool skip_special_tokens = true)
			{
				auto decodes_handle = tokenizers_decode_batch(*handle, &ids, ids.size(), skip_special_tokens, get_subarray_warp(ids));
				::rust::Vec* pdecode = reinterpret_cast<::rust::Vec*>(decodes_handle.ptr);
				std::vector<::rust::String> res;
				res.reserve(decodes_handle.len);
				for (size_t i = 0; i < decodes_handle.len; i++)
				{
					res.emplace_back(std::make_shared<::rust::SharedStringHandle>(pdecode[i]));
				}
				tokenizers_exported_strings_free_without_string_free(&decodes_handle);
				return res;
			}

			inline ::rust::String id_to_token(uint32_t id)
			{
				return ::rust::String(std::make_shared<::rust::SharedStringHandle>(tokenizers_id_to_token(*handle, id)));
			}

			inline uint32_t token_to_id(std::string_view token)
			{
				return tokenizers_token_to_id(*handle, token.data(), token.size());
			}

			inline size_t get_vocab_size()
			{
				return tokenizers_get_vocab_size(*handle);
			}

		private:
			std::shared_ptr<SharedTokenizerHandle> handle;
		};

	} // namespace rust_impl

} // namespace tokenizers

#include <mutex>
#include <unordered_map>
#include <optional>

namespace rust
{
	class HandlePool
	{
	public:
		struct Node
		{
			std::shared_ptr<interface::BaseSharedHandle> payload;
			size_t type = 0;
			std::_Atomic_counter_t counter = 0;

			inline std::optional<rust::String> string()
			{
				std::optional<rust::String> res = std::nullopt;
				if (type == typeid(rust::SharedStringHandle).hash_code())
				{
					res = rust::String(std::dynamic_pointer_cast<SharedStringHandle>(payload));
				}
				return res;
			}

			inline std::optional<tokenizers::rust_impl::Encoding> encoding()
			{
				std::optional<tokenizers::rust_impl::Encoding> res = std::nullopt;
				if (type == typeid(tokenizers::rust_impl::SharedEncodingHandle).hash_code())
				{
					res = tokenizers::rust_impl::Encoding(std::dynamic_pointer_cast<tokenizers::rust_impl::SharedEncodingHandle>(payload));
				}
				return res;
			}

			inline std::optional<std::vector<tokenizers::rust_impl::Encoding>> encodings()
			{
				std::optional<std::vector<tokenizers::rust_impl::Encoding>> res = std::nullopt;
				if (type == typeid(tokenizers::rust_impl::SharedEncodingArrayHandle).hash_code())
				{
					res = tokenizers::rust_impl::Encodings::fetch(std::dynamic_pointer_cast<tokenizers::rust_impl::SharedEncodingArrayHandle>(payload));
				}
				return res;
			}

			inline std::optional<tokenizers::rust_impl::Tokenizer> tokenizer()
			{
				std::optional<tokenizers::rust_impl::Tokenizer> res = std::nullopt;
				if (type == typeid(tokenizers::rust_impl::SharedTokenizerHandle).hash_code())
				{
					res = tokenizers::rust_impl::Tokenizer(std::dynamic_pointer_cast<tokenizers::rust_impl::SharedTokenizerHandle>(payload));
				}
				return res;
			}
		};

		inline Node& operator[](void* handle)
		{
			auto iter = lookup.find(handle);
			if (iter == lookup.end())
				return lookup[NULL];
			else
				return iter->second;
			;
		}

		template <class _Handle>
		inline void* register_handle(std::shared_ptr<_Handle> handle)
		{
			void* key = handle->handle();
			auto iter = lookup.find(key);
			if (iter != lookup.end())
			{
				_Incref(iter);
			}
			else
			{
				lookup.insert({ key, {handle, typeid(_Handle).hash_code(), 1} });
			}
			return key;
		}

		inline bool delete_handle(void* key)
		{
			auto iter = lookup.find(key);
			if (iter == lookup.end())
				return false;
			else
			{
				_Decref(iter);
				return true;
			}
		}

		inline std::optional<rust::String> to_string(void* handle)
		{
			return operator[](handle).string();
		}

		inline std::optional<tokenizers::rust_impl::Encoding> to_encoding(void* handle)
		{
			return operator[](handle).encoding();
		}

		inline std::optional<std::vector<tokenizers::rust_impl::Encoding>> to_encodings(void* handle)
		{
			return operator[](handle).encodings();
		}

		inline std::optional<tokenizers::rust_impl::Tokenizer> to_tokenizer(void* handle)
		{
			return operator[](handle).tokenizer();
		}

		static HandlePool& instance();
		static std::shared_ptr<HandlePool> instance_ptr();

		inline ~HandlePool()
		{
			lookup.clear();
		}

		inline void _Incref(std::_Atomic_counter_t count) noexcept { // increment use count
			_MT_INCR(count);
		}

		inline void _Incref(std::unordered_map<void*, Node>::iterator iter) noexcept { // increment use count
			_MT_INCR(iter->second.counter);
		}

		inline void _Decref(std::unordered_map<void*, Node>::iterator iter) noexcept { // decrement use count
			if (_MT_DECR(iter->second.counter) == 0) {
				lookup.erase(iter);
			}
		}

	private:
		inline HandlePool() : lookup()
		{
			lookup.insert({ NULL, {NULL, typeid(nullptr_t).hash_code()} });
		}

		std::unordered_map<void*, Node> lookup;

		static std::shared_ptr<HandlePool> pool;
		static std::mutex m_mutex;
	};

} // namespace rust

#endif // TOKENIZERS_RUST_H_