import tarfile
import lzma

def extract_and_merge(input, output, count, offset):
    with tarfile.open(input, 'r') as urlsf_tar:
        with open(output, 'wb') as output_file:
            for i in range(count):
                path = f"openwebtext/urlsf_subset09-{ offset + i + 1 }_data.xz"
                print(f"Extracting {path} to {output}")
                data_xz = urlsf_tar.extractfile(path)
                data_xz_bytes = data_xz.read()
                data_raw_bytes = lzma.decompress(data_xz_bytes)            
                output_file.write(data_raw_bytes)

extract_and_merge('data/urlsf_subset09.tar', 'data/webtext_train.txt', 998, 0)  # urlsf_subset09-[1,998]_data.xz
extract_and_merge('data/urlsf_subset09.tar', 'data/webtext_valid.txt', 1, 998)  # urlsf_subset09-999_data.xz
extract_and_merge('data/urlsf_subset09.tar', 'data/webtext_test.txt', 1, 999)   # urlsf_subset09-1000_data.xz
