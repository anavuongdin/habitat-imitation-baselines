import os

def reduce_query_handle(query_obj: str):
    query_obj = query_obj.split('/')[-1].split('_')[-1]
    return query_obj

def contains(src: str, query: str):
    return query in src

def is_glb_file(object_file: str):
    return contains(object_file, ".glb")

def make_objects_list():
    objects_path = 'data/test_assets/objects'
    objects_files = os.listdir(objects_path)
    objects = list(filter(is_glb_file, objects_files))

    return objects

def search(query_obj: str):
    objects = make_objects_list()
    query_obj = reduce_query_handle(query_obj)
    for obj in objects:
        if contains(obj, query_obj):
            return 'data/test_assets/objects/' + obj[:-4]
    return None


if __name__ == '__main__':
    query_obj = 'data/assets/objects/ycb_google_16k_v2/configs_gltf/024_bowl'
    print(search(query_obj))

