# tokenizers-cpp

This project provides a cross-platform C++ tokenizer binding library that can be universally deployed.
It wraps and binds the [HuggingFace tokenizers library](https://github.com/huggingface/tokenizers)
and [sentencepiece](https://github.com/google/sentencepiece) and provides a minimum common interface in C++.

The main goal of the project is to enable tokenizer deployment for language model applications
to native platforms with minimum dependencies and remove some of the barriers of
cross-language bindings. This project is developed in part with and
used in [MLC LLM](https://github.com/mlc-ai/mlc-llm). We have tested the following platforms:

- iOS
- Android
- Windows
- Linux
- Web browser

## Getting Started

The easiest way is to add this project as a submodule and then
include it via `add_sub_directory` in your CMake project.
You also need to turn on `c++17` support.

- First, you need to make sure you have rust installed.
- If you are cross-compiling make sure you install the necessary target in rust.
  For example, run `rustup target add aarch64-apple-ios` to install iOS target.
- You can then link the library

See [example](example) folder for an example CMake project.

### vcpkg

- Use vcpkg for dependency management.
- vcpkg installation can be referred from the [official documentation](https://learn.microsoft.com/zh-cn/vcpkg/get_started/get-started?pivots=shell-powershell).
- Update the vcpkg ports files using the files in the [vcpkg_modified](vcpkg_modified) folder to correctly compile.

### Example Code

```c++
// - dist/tokenizer.json
void HuggingFaceTokenizerExample1() {
  // Use Rust's interface to directly read the file.
  auto tok = tokenizers::Tokenizer::FromBlobJSONFile("dist/tokenizer.json");
  std::string_view prompt = "What is the capital of Canada?";
  // call Encode to turn prompt into struct Encoding.
  tokenizers::Encoding encoded = tok->Encode(prompt);
  // get token ids
  tokenizers::array_view<uint32_t> ids = encoded.ids.value();
  // call Decode to turn ids into struct Decoding
  tokenizers::Decoding decoded = tok->Decode(ids);
  // get decoded string
  std::string_view decoded_prompt = decoded.payload;
}

// or using memory file to get tokenizer
void HuggingFaceTokenizerExample2() {
  // Read blob from file.
  auto blob = LoadBytesFromFile("dist/tokenizer.json");
  // Note: all the current factory APIs takes in-memory blob as input.
  // This gives some flexibility on how these blobs can be read.
  auto tok = Tokenizer::FromBlobJSON(blob);
  ...
}

void SentencePieceTokenizerExample() {
  // Read blob from file.
  auto blob = LoadBytesFromFile("dist/tokenizer.model");
  // Note: all the current factory APIs takes in-memory blob as input.
  // This gives some flexibility on how these blobs can be read.
  auto tok = Tokenizer::FromBlobSentencePiece(blob);
  ...
}
```

### Extra Details

Currently, the project generates two static libraries on Linux
- `libtokenizers_c.a`: the c binding to tokenizers rust library
- `libtokenizers_cpp.a`: the cpp binding implementation

Two static libraries on Windows
- `tokenizers_c.lib`: the c binding to tokenizers rust library
- `tokenizers_cpp.lib`: the cpp binding implementation

If you are using an IDE, you can likely first use cmake to generate
these libraries and add them to your development environment.
If you are using cmake, `target_link_libraries(yourlib tokenizers_cpp)`
will automatically links in the other two libraries.
You can also checkout [MLC LLM](https://github.com/mlc-ai/mlc-llm)
for as an example of complete LLM chat application integrations.

## Javascript Support

We use emscripten to expose tokenizer-cpp to wasm and javascript.
Checkout [web](web) for more details.

## Acknowledgements

This project is only possible thanks to the shoulders open-source ecosystems that we stand on.
This project is based on sentencepiece and tokenizers library.
