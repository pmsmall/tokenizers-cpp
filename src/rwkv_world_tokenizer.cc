/*!
 *  Copyright (c) 2025 by Contributors
 * \file rwkv_world_tokenizer.cc
 * \brief Implementation of llm chat.
 */
#include "rwkv_world_tokenizer.h"

#include <tokenizers_cpp.h>

#include <fstream>
#include <msgpack.hpp>

namespace tokenizers
{

	struct TrieTree
	{
		std::unordered_map<int, std::unique_ptr<TrieTree>> children;
		std::string_view word;
		std::optional<int> token_id;

		TrieTree(const std::unordered_map<std::string_view, int>& word2id)
		{
			for (auto& pair : word2id)
			{
				add_word(pair.first, pair.second);
			}
		}

		std::pair<std::string, int> find_longest_prefix(std::string_view str) const
		{
			std::string prefix;
			int token_id = -1;
			const TrieTree* node = this;
			for (int i = 0; i < str.size(); ++i)
			{
				auto it = node->children.find(str[i]);
				if (it == node->children.end())
				{
					break;
				}
				node = it->second.get();
				RV_CHECK(node != nullptr);
				if (node->token_id.has_value())
				{
					prefix = node->word;
					token_id = node->token_id.value();
				}
			}
			RV_CHECK(!prefix.empty());
			RV_CHECK(token_id != -1);
			return { prefix, token_id };
		}

	private:
		TrieTree() = default;
		void add_word(std::string_view word, int token_id) { return _add_word(word, token_id, 0); }
		void _add_word(std::string_view word, int token_id, int idx)
		{
			if (idx == word.size())
			{
				this->word = word;
				this->token_id = token_id;
				return;
			}
			auto& child = children[word[idx]];
			if (!child)
			{
				child = std::unique_ptr<TrieTree>(new TrieTree());
			}
			child->_add_word(word, token_id, idx + 1);
		}
	};

	class RWKVWorldTokenizer : public Tokenizer
	{
	public:
		explicit RWKVWorldTokenizer(std::string_view path)
		{
			std::ifstream infile;
			infile.open(path.data(), std::ios::binary | std::ios::in);
			infile.seekg(0, std::ios::end);
			int64_t length = infile.tellg();
			infile.seekg(0, std::ios::beg);
			char* data = new char[length];
			infile.read(data, length);
			infile.close();

			auto unpacker = msgpack::unpack(data, length);
			auto obj = unpacker.get();
			delete[] data;
			_idx2word = obj.as<std::unordered_map<int, std::string>>();
			for (auto& pair : _idx2word)
			{
				_word2idx[pair.second] = pair.first;
			}
			_tree = std::make_unique<TrieTree>(_word2idx);
		}

		Encoding Encode(std::string_view str, bool add_special_tokens) final
		{
			std::shared_ptr<std::vector<uint32_t>> ids;
			size_t str_idx = 0;

			while (str_idx < str.size())
			{
				auto [prefix, token_id] = _tree->find_longest_prefix(str.substr(str_idx));
				ids->push_back(token_id);
				str_idx += prefix.size();
			}

			Encoding result = { {{.ids = array_view<uint32_t>(ids->data(), ids->size())}, {.payload = ids}} };

			return result;
		}

		Decoding Decode(array_view<uint32_t> ids, bool skip_special_tokens) final
		{
			std::string str;
			for (auto id : ids)
			{
				str += IdToToken(id).payload;
			}
			Decoding result = { {.buff = std::move(str)} };
			result.payload = result.buff.value();
			return result;
		}

		size_t GetVocabSize() final
		{
			auto size = _idx2word.size();
			RV_CHECK(size > 0);
			return size;
		}

		std::string_view GetToken(uint32_t token_id)
		{
			auto it = _idx2word.find(token_id);
			if (it == _idx2word.end())
			{
				return "<unk>";
			}
			else
			{
				return it->second;
			}
		}

		Decoding IdToToken(uint32_t token_id) final
		{
			RV_CHECK(_idx2word.size() > 0);
			return { .payload = GetToken(token_id) };
		}

		uint32_t TokenToId(std::string_view token) final
		{
			RV_CHECK(_word2idx.size() > 0);
			auto it = _word2idx.find(token);
			if (it == _word2idx.end())
			{
				return -1;
			}
			else
			{
				return it->second;
			}
		}

	private:
		// the tokenizer
		std::unordered_map<std::string_view, int> _word2idx;
		std::unordered_map<int, std::string> _idx2word;
		std::unique_ptr<TrieTree> _tree;
	};

	std::unique_ptr<Tokenizer> Tokenizer::FromBlobRWKVWorld(std::string_view model_blob)
	{
		return std::make_unique<RWKVWorldTokenizer>(model_blob);
	}

} // namespace tokenizers
