from visual_search.datasets import load_inshop, load_sop


def test_load_sop_uses_test_set_as_query_and_gallery(tmp_path):
    root = tmp_path / "Stanford_Online_Products"
    root.mkdir()
    (root / "Ebay_test.txt").write_text(
        "image_id class_id super_class_id path\n"
        "1 11 1 bike_final/111_0.JPG\n"
        "2 11 1 bike_final/111_1.JPG\n"
        "3 12 2 cup_final/121_0.JPG\n"
    )
    split = load_sop(tmp_path)
    assert split.gallery_labels.tolist() == [11, 11, 12]
    assert split.query_gallery_ids.tolist() == [0, 1, 2]
    assert split.gallery_paths[2] == root / "cup_final/121_0.JPG"


def test_load_inshop_separates_query_and_gallery(tmp_path):
    (tmp_path / "Eval").mkdir()
    (tmp_path / "Eval" / "list_eval_partition.txt").write_text(
        "4\nimage_name item_id evaluation_status\n"
        "img/a.jpg id_000001 train\n"
        "img/b.jpg id_000002 query\n"
        "img/c.jpg id_000002 gallery\n"
        "img/d.jpg id_000003 gallery\n"
    )
    split = load_inshop(tmp_path)
    assert [p.name for p in split.query_paths] == ["b.jpg"]
    assert split.gallery_labels.tolist() == [2, 3]
    assert split.query_gallery_ids is None
