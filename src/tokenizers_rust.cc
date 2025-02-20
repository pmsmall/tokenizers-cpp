/*!
 *  Copyright (c) 2025 by Contributors
 * \file tokenizers_rust.cc
 */
#include "tokenizers_rust.h"

void* tokenizers::alloc_string(size_t len, void* args)
{
	std::string* str = reinterpret_cast<std::string*>(args);
	str->resize(len + 1);
	//if (str->at(len)) str->at(len) = 0;
	return const_cast<char*>(str->data());
}

::rust::ArrayHandle tokenizers::fetch_string(void* arr, size_t offset)
{
	std::vector<std::string>* strs = reinterpret_cast<std::vector<std::string>*>(arr);
	std::string& str = strs->at(offset);
	return { str.data(),str.size() };
}

void* rust::SharedStringHandle::handle()
{
	return ptr;
}

void* tokenizers::rust_impl::SharedEncodingHandle::handle()
{
	return m_handle;
}

void* tokenizers::rust_impl::SharedEncodingArrayHandle::handle()
{
	return m_handle.ptr;
}

void* tokenizers::rust_impl::SharedTokenizerHandle::handle()
{
	return m_handle;
}

std::shared_ptr<rust::HandlePool> rust::HandlePool::pool;
std::mutex rust::HandlePool::m_mutex;

rust::HandlePool& rust::HandlePool::instance()
{
	return *instance_ptr();
}

std::shared_ptr<rust::HandlePool> rust::HandlePool::instance_ptr()
{
	if (!pool)
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		if (!pool)
		{
			pool = std::shared_ptr<rust::HandlePool>(new rust::HandlePool());
		}
	}

	return pool;
}
