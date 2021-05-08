import os
import requests
from tqdm import tqdm
from pydrive.auth import GoogleAuth
from run_utils import run_script_inline

def get_from_cache(dir_name: str, urls: dict, cache_dir: str = "~/.deepNeuroSegmenter"):    
    cache_dir = os.path.realpath(os.path.expanduser(cache_dir))
    print(cache_dir)
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    onlydirs = [f for f in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, dir_name))]
    print(onlydirs)
    if (not dir_name in onlydirs) or len(os.listdir(os.path.join(cache_dir, dir_name)))==0:
        deeper_dir = os.path.join(os.path.join(cache_dir, dir_name))
        if not os.path.exists(deeper_dir):
            os.mkdir(deeper_dir)
        for key in urls:
            filename = os.path.join(deeper_dir, key)
            file_id = urls[key]
            run_script_inline(f"""
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id={file_id}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {{print $NF}}' ./cookie`&id={file_id}" -o {filename}
""")








            '''
            deeper_dir = os.path.join(os.path.join(cache_dir, dir_name))
            if not os.path.exists(deeper_dir):
                os.mkdir(deeper_dir)
            print(deeper_dir)
            temp_filename = os.path.join(deeper_dir, key)
            print(temp_filename)
            # GET directory
            
            req = requests.get(urls[key], stream=True)
            print(urls[key])
            content_length = len(req.content)#req.headers.get("content-Length")
            print(len(req.content))
            
            total = int(content_length) if content_length is not None else None
            print(total)
            progress = tqdm(total)
            with open(temp_filename, "wb") as temp_file:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        progress.update(len(chunk))
                        temp_file.write(chunk)

            progress.close()
            

            with requests.get(urls[key], stream=True) as r:
                r.raise_for_status()
                with open(temp_filename, 'w') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        # If you have chunk encoded response uncomment if
                        # and set chunk_size parameter to None.
                        #if chunk: 
                        f.write(chunk)
            


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def main():
    



    

if __name__ == "__main__":
    main()
'''