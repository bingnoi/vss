id_map = {}
for cat in COCO_CATEGORIES_Seen:
    id_map[cat["id"]] = cat["trainId"]

def worker(file_tuple):
    file, output_file = file_tuple
    lab = np.asarray(Image.open(file))
    assert lab.dtype == np.uint8

    output = np.zeros_like(lab, dtype=np.uint8) + 255
    for obj_id in np.unique(lab):
        if obj_id in id_map:
            output[lab == obj_id] = id_map[obj_id]

    Image.fromarray(output).save(output_file)

if __name__ == "__main__":
    dataset_dir = "/datadisk/lixinhao" / Path(os.getenv("DETECTRON2_DATASETS", "dataset")) / "coco" / "coco_stuff"

    pool = Pool(32)

    for name in ["val2017_seen", "train2017"]:

        if name == "val2017_seen":
            annotation_dir = dataset_dir / "annotations" / "val2017"
        else:
            annotation_dir = dataset_dir / "annotations" / name
        output_dir = dataset_dir / "annotations_detectron2" / name
        output_dir.mkdir(parents=True, exist_ok=True)

        file_list = []
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            file_list.append((file, output_file))

        pool.map(worker, file_list)
        print('done {}'.format(name))