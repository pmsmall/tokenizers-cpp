// A simple C wrapper of tokenzier library
use serde_json::Value;
use std::{ collections::HashMap, ffi::c_void, mem, str::FromStr };
use tokenizers::{
    models::bpe::BPE,
    pre_tokenizers::byte_level::ByteLevel,
    tokenizer::{ Encoding, Tokenizer },
};

type CustomAllocatorArgs = *mut c_void;
type CustomAllocator = unsafe extern fn(usize, CustomAllocatorArgs) -> *mut c_void;
type CustomEmplaceBackArray = unsafe extern fn(CustomAllocatorArgs, *const c_void, usize);

#[repr(C)]
pub struct RustArrayHandle {
    ptr: *const c_void,
    len: usize,
}

// type CustomConvertArrayHandle = unsafe extern fn(*const c_void) -> RustArrayHandle;
type CustomConvertArrayHandleOffset = unsafe extern fn(*const c_void, usize) -> RustArrayHandle;

#[repr(C)]
pub struct ExportVec<T> {
    ptr: *const T,
    capacity: usize,
    len: usize,
    type_size: usize,
}

#[inline]
fn export_vec<T>(raw: Vec<T>) -> ExportVec<T> {
    let exported_vec: ExportVec<T> = ExportVec::<T> {
        ptr: raw.as_ptr(),
        capacity: raw.capacity(),
        len: raw.len(),
        type_size: mem::size_of::<T>(),
    };
    mem::forget(raw);
    return exported_vec;
}

#[inline]
fn export_string(raw: String) -> ExportVec<u8> {
    let exported_vec: ExportVec<u8> = ExportVec::<u8> {
        ptr: raw.as_ptr(),
        capacity: raw.capacity(),
        len: raw.len(),
        type_size: mem::size_of::<u8>(),
    };
    mem::forget(raw);
    return exported_vec;
}

pub type Vocab = HashMap<String, u32>;
pub type Merges = Vec<(String, String)>;

#[inline]
unsafe fn resize_cvec(
    len: usize,
    allocator: CustomAllocator,
    allocator_args: CustomAllocatorArgs
) -> *mut c_void {
    return allocator(len, allocator_args);
}

#[inline]
fn byte_level_bpe_from_str(vocab: &str, merges: &str, added_tokens: &str) -> Tokenizer {
    let vocab_json: Value = serde_json::from_str(vocab).unwrap();
    let added_tokens_json: Value = serde_json::from_str(added_tokens).unwrap();
    let mut vocab: HashMap<String, u32> = HashMap::new();
    match vocab_json {
        Value::Object(m) => {
            for (token, id) in m {
                if let Value::Number(id) = id {
                    let id = id.as_u64().unwrap() as u32;
                    vocab.insert(token, id);
                }
            }
        }
        _ => panic!("Invalid vocab.json file."),
    }
    match added_tokens_json {
        Value::Object(m) => {
            for (token, id) in m {
                if let Value::Number(id) = id {
                    let id = id.as_u64().unwrap() as u32;
                    vocab.insert(token, id);
                }
            }
        }
        _ => panic!("Invalid added_tokens.json file."),
    }

    let merges: Vec<(String, String)> = merges
        .lines()
        .filter(|line| !line.starts_with("#version"))
        .map(|line| {
            let parts = line.split(' ').collect::<Vec<_>>();
            if parts.len() != 2 {
                panic!("Invalid merges.txt file.");
            }
            return (parts[0].to_string(), parts[1].to_string()); // Add the `return` keyword here
        })
        .collect::<Vec<(String, String)>>();
    let byte_level = ByteLevel::new(false, false, false);
    let mut tokenizer: Tokenizer = Tokenizer::new(BPE::new(vocab, merges));
    tokenizer.with_pre_tokenizer(Some(byte_level)).with_decoder(Some(byte_level));
    return tokenizer;
}

#[no_mangle]
extern "C" fn tokenizers_new_from_str(input_cstr: *const u8, len: usize) -> *mut Tokenizer {
    unsafe {
        let json: &str = std::str::from_utf8(std::slice::from_raw_parts(input_cstr, len)).unwrap();
        return Box::into_raw(Box::new(Tokenizer::from_str(json).unwrap().into()));
    }
}

#[no_mangle]
extern "C" fn tokenizers_new_from_file(input_cstr: *const u8, len: usize) -> *mut Tokenizer {
    unsafe {
        let path: &str = std::str::from_utf8(std::slice::from_raw_parts(input_cstr, len)).unwrap();
        return Box::into_raw(Box::new(Tokenizer::from_file(path)).unwrap().into());
    }
}

#[no_mangle]
extern "C" fn tokenizers_new_from_byte_level_bpe(
    input_vocab_str: *const u8,
    len_vocab: usize,
    input_merges_str: *const u8,
    len_merges: usize,
    input_added_tokens_str: *const u8,
    len_added_tokens: usize
) -> *mut Tokenizer {
    unsafe {
        let vocab: &str = std::str
            ::from_utf8(std::slice::from_raw_parts(input_vocab_str, len_vocab))
            .unwrap();
        let merges: &str = std::str
            ::from_utf8(std::slice::from_raw_parts(input_merges_str, len_merges))
            .unwrap();
        let added_tokens: &str = std::str
            ::from_utf8(std::slice::from_raw_parts(input_added_tokens_str, len_added_tokens))
            .unwrap();
        return Box::into_raw(Box::new(byte_level_bpe_from_str(vocab, merges, added_tokens)));
    }
}

#[no_mangle]
extern "C" fn tokenizers_encode(
    handle: *mut Tokenizer,
    input_cstr: *const u8,
    len: usize,
    add_special_tokens: i32
) -> *mut Encoding {
    unsafe {
        let input_data: &str = std::str
            ::from_utf8(std::slice::from_raw_parts(input_cstr, len))
            .unwrap();
        return Box::into_raw(
            Box::new((*handle).encode(input_data, add_special_tokens != 0).unwrap() as Encoding)
        );
    }
}

#[no_mangle]
extern "C" fn tokenizers_encoding_ids(encoding_handle: *mut Encoding) -> RustArrayHandle {
    unsafe {
        let ids: &[u32] = (*encoding_handle).get_ids();
        let len: usize = ids.len();
        return RustArrayHandle { ptr: ids.as_ptr().cast(), len: len };
    }
}

#[no_mangle]
extern "C" fn tokenizers_encoding_type_ids(encoding_handle: *mut Encoding) -> RustArrayHandle {
    unsafe {
        let type_ids: &[u32] = (*encoding_handle).get_type_ids();
        let len: usize = type_ids.len();
        return RustArrayHandle { ptr: type_ids.as_ptr().cast(), len: len };
    }
}

#[no_mangle]
extern "C" fn tokenizers_encoding_tokens(
    encoding_handle: *mut Encoding,
    allocator: CustomAllocator,
    allocator_args: CustomAllocatorArgs,
    emplace_back: CustomEmplaceBackArray
) {
    unsafe {
        let tokens: &[String] = (*encoding_handle).get_tokens();
        let len: usize = tokens.len();
        resize_cvec(len, allocator, allocator_args);
        tokens.iter().for_each(|n_string| {
            emplace_back(allocator_args, n_string.as_str().as_ptr().cast(), n_string.len());
        });
    }
}

#[no_mangle]
extern "C" fn tokenizers_encoding_special_tokens_mask(
    encoding_handle: *mut Encoding
) -> RustArrayHandle {
    unsafe {
        let special_tokens_mask: &[u32] = (*encoding_handle).get_special_tokens_mask();
        let len: usize = special_tokens_mask.len();
        return RustArrayHandle { ptr: special_tokens_mask.as_ptr().cast(), len: len };
    }
}

#[no_mangle]
extern "C" fn tokenizers_encoding_attention_mask(
    encoding_handle: *mut Encoding
) -> RustArrayHandle {
    unsafe {
        let attention_mask: &[u32] = (*encoding_handle).get_attention_mask();
        let len: usize = attention_mask.len();
        return RustArrayHandle { ptr: attention_mask.as_ptr().cast(), len: len };
    }
}

#[no_mangle]
extern "C" fn tokenizers_encode_batch(
    handle: *mut Tokenizer,
    input_cstr: *const c_void,
    num_seqs: usize,
    add_special_tokens: i32,
    convert_array_offset: CustomConvertArrayHandleOffset
) -> ExportVec<Encoding> {
    unsafe {
        let input_data: Vec<&str> = (0..num_seqs)
            .map(|i: usize| {
                let array_handle = convert_array_offset(input_cstr, i);
                std::str
                    ::from_utf8(
                        std::slice::from_raw_parts(array_handle.ptr as *const u8, array_handle.len)
                    )
                    .unwrap()
            })
            .collect::<Vec<&str>>();
        let encodings: Vec<Encoding> = (*handle)
            .encode_batch(input_data, add_special_tokens != 0)
            .unwrap();

        return export_vec(encodings);
    }
}

#[no_mangle]
extern "C" fn tokenizers_decode(
    handle: *mut Tokenizer,
    input_ids: *const u32,
    len: usize,
    skip_special_tokens: i32
) -> ExportVec<u8> {
    unsafe {
        let input_data: &[u32] = std::slice::from_raw_parts(input_ids, len);

        return export_string((*handle).decode(input_data, skip_special_tokens != 0).unwrap());
    }
}

#[no_mangle]
extern "C" fn tokenizers_decode_batch(
    handle: *mut Tokenizer,
    input_ids: *const c_void,
    raws: usize,
    skip_special_tokens: i32,
    convert_array_offset: CustomConvertArrayHandleOffset
) -> ExportVec<ExportVec<u8>> {
    unsafe {
        let input_data: Vec<&[u32]> = (0..raws)
            .map(|i: usize| {
                let array_handle = convert_array_offset(input_ids, i);
                std::slice::from_raw_parts(array_handle.ptr as *const u32, array_handle.len)
            })
            .collect::<Vec<&[u32]>>();

        return export_vec(
            (*handle)
                .decode_batch(&input_data, skip_special_tokens != 0)
                .unwrap()
                .into_iter()
                .map(|s| { export_string(s) })
                .collect::<Vec<ExportVec<u8>>>()
        );
    }
}

#[no_mangle]
extern "C" fn tokenizers_get_vocab_size(handle: *mut Tokenizer) -> usize {
    unsafe {
        return (*handle).get_vocab_size(true);
    }
}

#[no_mangle]
extern "C" fn tokenizers_id_to_token(handle: *mut Tokenizer, id: u32) -> ExportVec<u8> {
    unsafe {
        let str: String = (*handle).id_to_token(id).unwrap();
        return export_string(str);
    }
}

#[no_mangle]
extern "C" fn tokenizers_token_to_id(handle: *mut Tokenizer, ctoken: *const u8, len: usize) -> u32 {
    unsafe {
        let token = std::str::from_utf8(std::slice::from_raw_parts(ctoken, len)).unwrap();
        return (*handle).token_to_id(token).unwrap();
    }
}

#[no_mangle]
extern "C" fn tokenizers_free(handle: *mut Tokenizer) {
    unsafe {
        mem::drop(Box::from_raw(handle));
    }
}

#[no_mangle]
extern "C" fn tokenizers_encoding_free(handle: *mut Encoding) {
    unsafe {
        mem::drop(Box::from_raw(handle));
    }
}

#[no_mangle]
extern "C" fn tokenizers_encodings_free(handle: *mut ExportVec<Encoding>) {
    unsafe {
        let encodings = Vec::from_raw_parts(
            (*handle).ptr as *mut Encoding,
            (*handle).len,
            (*handle).capacity
        );
        mem::drop(encodings);
    }
}

#[no_mangle]
extern "C" fn tokenizers_exported_string_free(handle: *mut ExportVec<u8>) {
    unsafe {
        let exported_string = Vec::from_raw_parts(
            (*handle).ptr as *mut u8,
            (*handle).len,
            (*handle).capacity
        );
        mem::drop(exported_string);
    }
}

#[no_mangle]
extern "C" fn tokenizers_exported_string_free_with_args(ptr: *mut u8, len: usize, capacity: usize) {
    unsafe {
        let exported_string = Vec::from_raw_parts(ptr, len, capacity);
        mem::drop(exported_string);
    }
}

#[no_mangle]
extern "C" fn tokenizers_exported_strings_free(handle: *mut ExportVec<ExportVec<u8>>) {
    unsafe {
        let exported_strings = Vec::from_raw_parts(
            (*handle).ptr as *mut ExportVec<u8>,
            (*handle).len,
            (*handle).capacity
        );

        exported_strings.iter().for_each(|s| {
            let exported_string = Vec::from_raw_parts(s.ptr as *mut u8, s.len, s.capacity);
            mem::drop(exported_string);
        });

        mem::drop(exported_strings);
    }
}


#[no_mangle]
extern "C" fn tokenizers_exported_strings_free_without_string_free(handle: *mut ExportVec<ExportVec<u8>>) {
    unsafe {
        let exported_strings = Vec::from_raw_parts(
            (*handle).ptr as *mut ExportVec<u8>,
            (*handle).len,
            (*handle).capacity
        );

        mem::drop(exported_strings);
    }
}
