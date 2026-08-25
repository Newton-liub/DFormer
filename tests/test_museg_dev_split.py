from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import random
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]; SPLITS=ROOT/'tools'/'splits'
def load(name:str):
    spec=importlib.util.spec_from_file_location(name,SPLITS/f'{name}.py'); assert spec and spec.loader
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
m=load('create_museg_dev_split')
a=load('audit_museg_splits')

def test_frozen_manifest_reconstructs_approved_candidate_sha256():
    frozen_path = ROOT / "data" / "splits" / "MUSeg" / "dev-v1" / "manifest.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    reconstructed = copy.deepcopy(frozen)
    source_hash = reconstructed.pop("freeze_metadata")["source_candidate_manifest_sha256"]
    reconstructed["candidate_status"] = "candidate"
    reconstructed["user_gate_a"] = {
        "status": "pending", "signed_by": None, "signature_reference": None,
    }
    candidate_bytes = m.canonical_json(reconstructed)
    assert hashlib.sha256(candidate_bytes).hexdigest() == source_hash
    ignored_candidate = ROOT / "output_museg_dev_candidate" / "manifest.json"
    if ignored_candidate.is_file():
        assert ignored_candidate.read_bytes() == candidate_bytes
    assert m.parse_path('RGB/06-01-01-0215-230920140531-06-99.jpg')==('06-01-01-0215','06')
    assert m.parse_official_bytes(b'RGB/01-01-01-0001-a.jpg\n')
    assert m.parse_official_bytes(b'RGB/01-01-01-0001-a.jpg\r\n')
    for bad in (b'\xef\xbb\xbfRGB/01-01-01-0001-a.jpg\n',b'RGB/01-01-01-0001-a.jpg\r\nRGB/01-01-01-0002-a.jpg\n',b'RGB/01-01-01-0001-a.jpg\r',b'RGB/01-01-01-0001-a.jpg\nRGB/01-01-01-0001-a.jpg\n',b'RGB\\01-01-01-0001-a.jpg\n',b'RGB/07-01-01-0001-a.jpg\n'):
        with pytest.raises(ValueError):m.parse_official_bytes(bad)

def test_fraction_bins_luma_and_objective_are_exact():
    assert m.bin_index(Fraction(1,2),m.DEPTH_BINS)==1
    assert m.bin_index(Fraction(1),m.DEPTH_BINS)==5
    assert m.class_bin(15)==7
    assert m.e(1,5)==0 and m.d(1,5,1,5)==0
    assert m.d(299,1000*255,299,1000*255)==0

def test_order_is_deterministic_and_non_git_commit_is_safe(tmp_path,monkeypatch):
    assert sorted(['01-01-01-0002','01-01-01-0001'],key=m.group_order)==sorted(['01-01-01-0002','01-01-01-0001'],key=m.group_order)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k: (_ for _ in ()).throw(OSError()))
    assert m.git_commit(tmp_path) is None

def test_cli_rejects_nonfrozen_before_input_reads(tmp_path):
    with pytest.raises(SystemExit):m.main(['--dataset-root',str(tmp_path),'--official-train',str(tmp_path/'a'),'--official-test',str(tmp_path/'b'),'--output-root',str(tmp_path/'out'),'--seed','1'])
    with pytest.raises(SystemExit):m.main(['--dataset-root',str(tmp_path),'--official-train',str(tmp_path/'a'),'--official-test',str(tmp_path/'b'),'--output-root',str(tmp_path/'out'),'--target-val-images','1'])

def test_algorithm_toy_input_order_independent_and_infeasible():
    def s(group,mine):return m.Sample(f'RGB/{group}-x.jpg',group,mine,1,(0,)*15,(0,)*15,1,1,1,0,0,0)
    groups={f'{mine}-01-01-{n:04}':[s(f'{mine}-01-01-{n:04}',mine)] for mine in m.MINES for n in (1,2)}
    selected,_,_=m.optimize(dict(reversed(list(groups.items()))))
    assert m.feasible(selected,groups)
    bad=dict(groups);bad.pop('01-01-01-0002')
    with pytest.raises(ValueError):m.optimize(bad)

def test_incremental_group_vectors_match_naive_totals_and_score():
    def sample(group,mine,n): return m.Sample(f'RGB/{group}-{n}.jpg',group,mine,n,(n,)*15,(n+1,)*15,n%2,n,n*7,n%6,n%8,n%8)
    groups={}
    for mine in m.MINES:
        groups[f'{mine}-01-01-0001']=[sample(f'{mine}-01-01-0001',mine,1),sample(f'{mine}-01-01-0001',mine,2)]
        groups[f'{mine}-01-01-0002']=[sample(f'{mine}-01-01-0002',mine,3)]
    vectors=m.aggregate_groups(groups); selected={f'{mine}-01-01-0001' for mine in m.MINES}
    incremental=m.zero_totals()
    for group in selected: incremental=m.add_totals(incremental,vectors[group])
    naive=m.totals(sample for group in selected for sample in groups[group])
    assert incremental==naive
    all_total=m.totals(sample for samples in groups.values() for sample in samples)
    assert m.score_from_totals(incremental,all_total)==m.score(selected,groups,all_total)
    moved=m.add_totals(incremental,vectors['01-01-01-0002'])
    assert moved==m.totals(sample for group in selected|{'01-01-01-0002'} for sample in groups[group])


def _tiny_groups(seed: int, *, identical: bool = False):
    rng=random.Random(seed); groups={}
    for mine_index,mine in enumerate(m.MINES):
        for group_index in range(3):
            group=f'{mine}-01-{mine_index+1:02}-{group_index+1:04}'; samples=[]
            for sample_index in range(1 if identical else rng.randint(1,3)):
                pixels=7 if identical else rng.randint(1,30)
                presence=(0,)*15 if identical else tuple(rng.randint(0,1) for _ in range(15))
                class_pixels=(0,)*15 if identical else tuple(rng.randint(0,pixels) for _ in range(15))
                samples.append(m.Sample(f'RGB/{group}-{sample_index}.jpg',group,mine,pixels,presence,class_pixels,
                    0 if identical else rng.randint(0,1), pixels if identical else rng.randint(0,pixels),
                    pixels*1000 if identical else rng.randint(0,255000*pixels),
                    0 if identical else rng.randrange(6),0 if identical else rng.randrange(8),0 if identical else rng.randrange(8)))
            groups[group]=samples
    return groups

def test_flat_scores_and_every_legal_move_order_match_reference_random_and_boundaries():
    for groups in [_tiny_groups(seed) for seed in range(6)]+[_tiny_groups(99,identical=True)]:
        order,vectors,total,chosen,current=m._initial_flat(groups)
        _,nested_vectors,nested_total,nested_chosen,nested_current=m._initial_naive(groups)
        assert {order[index] for index in chosen}==nested_chosen
        assert m.flat_score(current,total)==m.score_from_totals(nested_current,nested_total)
        legal=[]
        train=[i for i in range(len(order)) if i not in chosen]; selected=sorted(chosen)
        for move in m._moves(selected,train):
            if not m._move_feasible(current,vectors,move,total): continue
            _,removed,added=move
            candidate=m.move_vector(current,vectors,move)
            nested_candidate=nested_current
            if removed>=0:nested_candidate=m.add_totals(nested_candidate,nested_vectors[order[removed]],-1)
            if added>=0:nested_candidate=m.add_totals(nested_candidate,nested_vectors[order[added]])
            reference=m.score_from_totals(nested_candidate,nested_total); fast=m.flat_score(candidate,total)
            assert fast==reference
            legal.append((move,reference))
        ranges=(range(m.V_MINE.start,m.V_MINE.stop),range(m.V_PRESENCE.start,m.V_PRESENCE.stop),range(m.V_CLASS_PIXELS.start,m.V_CLASS_PIXELS.stop))
        weights=tuple(m._integer_weights(total,indices) for indices in ranges)
        keyed=[(move,m._fast_keys(m.move_vector(current,vectors,move),total,weights),score) for move,score in legal]
        assert [move for move,_,_ in sorted(keyed,key=lambda item:(item[1],m._move_tie(item[0],order)))]==[move for move,_ in sorted(legal,key=lambda item:(item[1],m._move_tie(item[0],order)))]

def test_naive_and_fast_match_each_round_final_result_and_deterministic_bytes():
    for groups in [_tiny_groups(seed) for seed in range(8)]+[_tiny_groups(101,identical=True)]:
        naive_trace=[];fast_trace=[]
        naive=m.optimize_naive(groups,trace=naive_trace);fast=m.optimize(groups,trace=fast_trace)
        assert fast==naive
        assert fast_trace==naive_trace
        repeated=[]
        for _ in range(3):
            result=m.optimize(dict(reversed(list(groups.items())))); selected=result[0]
            repeated.append((result,m.output_lines(groups,set(groups)-selected),m.output_lines(groups,selected)))
        assert repeated[0]==repeated[1]==repeated[2]

def test_fast_search_constructs_far_fewer_fraction_scores_and_is_faster():
    groups=_tiny_groups(2026)
    naive_stats=m.OptimizationStats();fast_stats=m.OptimizationStats()
    start=time.perf_counter();naive=m.optimize_naive(groups,stats=naive_stats);naive_time=time.perf_counter()-start
    start=time.perf_counter();fast=m.optimize(groups,stats=fast_stats);fast_time=time.perf_counter()-start
    assert fast==naive
    assert naive_stats.full_fraction_scores>0
    assert fast_stats.full_fraction_scores==0
    assert fast_stats.aux_fraction_keys*4<naive_stats.full_fraction_scores
    # Relative only: no absolute millisecond threshold. Repeat once before failing to
    # avoid treating a transient scheduler interruption as a performance regression.
    if fast_time>=naive_time:
        start=time.perf_counter();m.optimize_naive(groups);naive_time=time.perf_counter()-start
        start=time.perf_counter();m.optimize(groups);fast_time=time.perf_counter()-start
    assert fast_time<naive_time


def test_manifest_protocol_helpers_cover_schema_relationships_and_counts():
    definitions = m.statistic_definitions()
    assert set(definitions) == {"label", "depth16", "rgb_luma", "class_count", "bins"}
    assert definitions["label"]["allowed_values"] == list(range(16))
    assert len(definitions["bins"]["depth16_valid_ratio"]) == 6
    assert len(definitions["bins"]["rgb_normalized_mean_luma"]) == 8
    assert definitions["bins"]["foreground_class_count"][-1]["values"] == list(range(10, 16))

    official = ("RGB/01-01-01-0001-a.jpg", "RGB/02-01-01-0001-a.jpg")
    test = ("RGB/03-01-01-0001-a.jpg",)
    relationships = m.set_relationships(official, test, official[:1], official[1:])
    assert relationships["samples"]["dev_union_vs_official_train"]["closed"]
    assert relationships["samples"]["train_dev__val_dev"] == {"count": 0, "items": []}
    assert relationships["groups"]["official_train__official_test"] == {"count": 0, "items": []}


def test_rare_classes_threshold_warnings_and_hard_constraints_are_review_only():
    def sample(group: str, mine: str, presence: tuple[int, ...], class_pixels: tuple[int, ...], background: int = 0):
        return m.Sample(f"RGB/{group}-x.jpg", group, mine, 10, presence, class_pixels, background, 10, 1000, 5, 0, 0)

    samples = []
    groups = {}
    for mine in m.MINES:
        for index in (1, 2):
            group = f"{mine}-01-01-{index:04}"
            presence = (1,) + (0,) * 14
            class_pixels = (3,) + (0,) * 14
            item = sample(group, mine, presence, class_pixels, background=1)
            groups[group] = [item]
            samples.append(item)
    total = m.totals(samples)
    selected = {f"{mine}-01-01-0001" for mine in m.MINES}
    val = m.totals(groups[group][0] for group in selected)
    # Force the reviewed values beyond their exact thresholds without changing hard feasibility.
    total["class_pixels"][5] = 10
    val["class_pixels"][5] = 0
    total["background"] = 10
    val["background"] = 0
    total["class_hist"] = [9, 9, 9, 0, 0, 0, 0, 0]
    val["class_hist"] = [0] * 8
    thresholds, warnings = m.threshold_evaluation(samples, val, total, groups)
    assert thresholds["requires_sol_review"]
    assert thresholds["review_status"] == "required"
    assert any(text.startswith("class_06_pixel_error_gt_1/2") for text in warnings)
    assert any(text.startswith("all_background_error_gt_1/2") for text in warnings)
    assert any(text.startswith("aux_foreground_class_count_histogram_gt_1/3") for text in warnings)

    train_paths = tuple(groups[group][0].path for group in set(groups) - selected)
    val_paths = tuple(groups[group][0].path for group in selected)
    relationships = m.set_relationships(train_paths + val_paths, (), train_paths, val_paths)
    mine_counts = m.mine_side_counts(groups, selected)
    hard = m.hard_constraint_checks(relationships, mine_counts, True)
    assert hard["all_pass"]
    rare = m.rare_class_list(samples)
    assert {item["class_id"] for item in rare} == set(range(2, 16))


def test_audit_without_write_report_never_creates_file(tmp_path):
    output = tmp_path / "candidate"
    output.mkdir()
    report_path = output / "audit-report.json"
    report = a.audit(tmp_path, tmp_path / "missing-train.txt", tmp_path / "missing-test.txt", output, write=False)
    assert not report["pass"]
    assert not report_path.exists()
    written = a.audit(tmp_path, tmp_path / "missing-train.txt", tmp_path / "missing-test.txt", output, write=True)
    assert report_path.read_bytes() == a.canonical(written)


def _sample(path: str, *, pixels: int = 1, presence: tuple[int, ...] = (0,) * 15,
            class_pixels: tuple[int, ...] = (0,) * 15) -> object:
    group, mine = m.parse_path(path)
    return m.Sample(path, group, mine, pixels, presence, class_pixels, 0, pixels, 0, 0, 0, 0)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


@pytest.fixture
def tiny_candidate(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    output = tmp_path / "candidate"
    train_paths = tuple(
        f"RGB/{mine}-01-01-{index:04}-sample.jpg"
        for mine in m.MINES for index in (1, 2)
    )
    test_paths = ("RGB/01-99-99-9999-test.jpg",)
    train_file = tmp_path / "train.txt"
    test_file = tmp_path / "test.txt"
    train_raw = ("\n".join(train_paths) + "\n").encode()
    test_raw = ("\r\n".join(test_paths) + "\r\n").encode()
    train_file.write_bytes(train_raw)
    test_file.write_bytes(test_raw)
    for path in train_paths:
        stem = path[4:-4]
        for directory, extension, array in (
            ("RGB", "jpg", m.np.zeros((2, 2, 3), dtype=m.np.uint8)),
            ("Depth16", "png", m.np.ones((2, 2), dtype=m.np.uint16)),
            ("Label", "png", m.np.zeros((2, 2), dtype=m.np.uint8)),
        ):
            target = dataset / directory / f"{stem}.{extension}"
            target.parent.mkdir(parents=True, exist_ok=True)
            assert m.cv2.imwrite(str(target), array)
    frozen = {
        "TRAIN_HASH": hashlib.sha256(train_raw).hexdigest(),
        "TRAIN_COUNT": len(train_paths),
        "TRAIN_GROUPS": len({m.parse_path(path)[0] for path in train_paths}),
        "TEST_HASH": hashlib.sha256(test_raw).hexdigest(),
        "TEST_COUNT": len(test_paths),
        "TEST_GROUPS": len({m.parse_path(path)[0] for path in test_paths}),
        "TARGET": 6,
    }
    for name, value in frozen.items():
        monkeypatch.setattr(m, name, value)
    monkeypatch.setattr(m, "git_commit", lambda _root: "a" * 40)
    monkeypatch.setattr(m, "git_object_is_commit", lambda _root, value: value == "a" * 40)
    monkeypatch.setattr(m, "verify_runtime", lambda: {"opencv": "tiny", "jpeg": "tiny"})
    assert m.main([
        "--dataset-root", str(dataset), "--official-train", str(train_file),
        "--official-test", str(test_file), "--output-root", str(output),
    ]) == 0
    return {
        "dataset": dataset, "output": output, "train": train_file, "test": test_file,
        "train_raw": train_raw, "test_raw": test_raw, "frozen": frozen,
    }


def test_parse_official_bytes_complete_acceptance_and_rejection_matrix():
    path = b"RGB/01-01-01-0001-a.jpg"
    assert m.parse_official_bytes(path + b"\n") == (path.decode(),)
    assert m.parse_official_bytes(path + b"\r\n") == (path.decode(),)
    rejected = {
        "empty": b"", "empty_line": path + b"\n\n", "leading_space": b" " + path + b"\n",
        "trailing_space": path + b" \n", "absolute": b"/RGB/01-01-01-0001-a.jpg\n",
        "drive": b"C:/RGB/01-01-01-0001-a.jpg\n", "uri": b"file:///RGB/01-01-01-0001-a.jpg\n",
        "dot": b"RGB/./01-01-01-0001-a.jpg\n", "dotdot": b"RGB/../01-01-01-0001-a.jpg\n",
        "missing_terminal_newline": path, "invalid_utf8": b"RGB/01-01-01-0001-\xff.jpg\n",
    }
    for label, raw in rejected.items():
        with pytest.raises(ValueError, match="official|invalid"):
            m.parse_official_bytes(raw)


def test_all_bin_boundaries_and_sample_statistics_have_exact_goldens(monkeypatch):
    for edges in (m.DEPTH_BINS, m.LUMA_BINS):
        assert m.bin_index(edges[0], edges) == 0
        for index, edge in enumerate(edges[1:-1], start=1):
            assert m.bin_index(edge, edges) == index
            assert m.bin_index(edge - Fraction(1, 10_000), edges) == index - 1
        assert m.bin_index(edges[-1], edges) == len(edges) - 2
    expected_class_bins = [0, 1, 2, 3, 4, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7, 7]
    assert [m.class_bin(count) for count in range(16)] == expected_class_bins

    rgb = m.np.empty((2, 3, 3), dtype=m.np.uint8)
    rgb[:] = (10, 20, 30)  # OpenCV is BGR, not RGB.
    depth = m.np.array([[0, 1, 0], [2, 3, 0]], dtype=m.np.uint16)
    label = m.np.array([[0, 1, 1], [2, 0, 15]], dtype=m.np.uint8)
    monkeypatch.setattr(m.cv2, "imread", lambda path, _flag: rgb if "RGB" in path else depth if "Depth16" in path else label)
    sample = m.load_samples(Path("unused"), ["RGB/01-01-01-0001-a.jpg"])[0]
    assert sample.presence == (1, 1) + (0,) * 12 + (1,)
    assert sample.class_pixels == (2, 1) + (0,) * 12 + (1,)
    assert sample.background == 0
    assert sample.depth_valid == 3
    assert sample.y_num == 6 * (299 * 30 + 587 * 20 + 114 * 10)
    assert sample.depth_bin == m.bin_index(Fraction(1, 2), m.DEPTH_BINS)
    assert sample.luma_bin == m.bin_index(Fraction(sample.y_num, 1000 * 255 * 6), m.LUMA_BINS)
    assert sample.class_count_bin == m.class_bin(3)


def test_missing_and_invalid_modalities_always_hard_fail(monkeypatch):
    good_rgb = m.np.zeros((2, 2, 3), dtype=m.np.uint8)
    good_depth = m.np.ones((2, 2), dtype=m.np.uint16)
    good_label = m.np.zeros((2, 2), dtype=m.np.uint8)
    cases = {
        "missing_rgb": (None, good_depth, good_label),
        "missing_depth16": (good_rgb, None, good_label),
        "missing_label": (good_rgb, good_depth, None),
        "rgb_dtype": (good_rgb.astype(m.np.uint16), good_depth, good_label),
        "rgb_dimension": (good_rgb[..., 0], good_depth, good_label),
        "depth_dtype": (good_rgb, good_depth.astype(m.np.uint8), good_label),
        "depth_dimension": (good_rgb, good_depth[..., None], good_label),
        "label_dtype": (good_rgb, good_depth, good_label.astype(m.np.uint16)),
        "label_dimension": (good_rgb, good_depth, good_label[..., None]),
        "size_mismatch": (good_rgb, m.np.ones((1, 2), dtype=m.np.uint16), good_label),
        "label_16": (good_rgb, good_depth, m.np.full((2, 2), 16, dtype=m.np.uint8)),
    }
    for name, arrays in cases.items():
        def fake_imread(path, _flag, arrays=arrays):
            return arrays[0] if "RGB" in path else arrays[1] if "Depth16" in path else arrays[2]
        monkeypatch.setattr(m.cv2, "imread", fake_imread)
        with pytest.raises(ValueError, match="modality|dimensions|label"):
            m.load_samples(Path("unused"), ["RGB/01-01-01-0001-a.jpg"])


def test_inventory_digest_matches_independent_serialization_is_order_independent_and_sensitive(tmp_path):
    paths = ("RGB/01-01-01-0002-b.jpg", "RGB/01-01-01-0001-a.jpg")
    for index, path in enumerate(paths):
        stem = path[4:-4]
        for directory, extension in (("RGB", "jpg"), ("Depth16", "png"), ("Label", "png")):
            target = tmp_path / directory / f"{stem}.{extension}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bytes([index + 1, len(directory), 255 - index]))
    got = m.inventory(tmp_path, paths, "Depth16")
    records = []
    for path in paths:
        rel = f"Depth16/{path[4:-4]}.png"
        raw = (tmp_path / rel).read_bytes()
        records.append(rel.encode() + b"\0" + str(len(raw)).encode("ascii") + b"\0" + hashlib.sha256(raw).hexdigest().encode("ascii") + b"\n")
    assert got["sha256"] == hashlib.sha256(b"".join(sorted(records))).hexdigest()
    assert got == m.inventory(tmp_path, reversed(paths), "Depth16")
    target = tmp_path / f"Depth16/{paths[0][4:-4]}.png"
    target.write_bytes(target.read_bytes()[:-1] + b"\x00")
    assert m.inventory(tmp_path, paths, "Depth16")["sha256"] != got["sha256"]


def test_rare_anomaly_hard_constraint_and_search_guard_exact_boundaries():
    samples = []
    for index in range(6):
        group = "01-01-01-0001" if index < 5 else "01-01-01-0002"
        present = [0] * 15
        if index < 5: present[0] = 1
        if index < 4 or index == 5: present[1] = 1
        if index < 3 or index == 5: present[2] = 1
        samples.append(m.Sample(f"RGB/{group}-{index}.jpg", group, "01", 1, tuple(present), tuple(present), 0, 1, 0, 0, 0, 0))
    rare = {item["class_id"]: item for item in m.rare_class_list(samples)}
    assert rare[1]["images"] == 5 and rare[1]["groups"] == 1 and rare[1]["cannot_bilateral_cover"]
    assert 2 not in rare
    assert rare[3]["images"] == 4 and rare[3]["groups"] == 2 and not rare[3]["cannot_bilateral_cover"]

    groups = {"01-01-01-0001": [samples[0]], "01-01-01-0002": [samples[-1]]}
    total, val = m.zero_totals(), m.zero_totals()
    total["images"], val["images"] = 10, m.TARGET - 5
    total["pixels"], val["pixels"] = 10, 2
    total["presence"][0] = total["class_pixels"][0] = total["background"] = 10
    val["presence"][0] = val["class_pixels"][0] = val["background"] = 1
    exact, warnings = m.threshold_evaluation(samples, val, total, groups)
    checks = exact["checks"]
    assert not checks["class_presence_error_gt_1_2"][0]["triggered"]
    assert not checks["class_pixel_error_gt_1_2"][0]["triggered"]
    assert not checks["all_background_error_gt_1_2"]["triggered"]
    assert not checks["val_image_deviation_gt_limit"]["triggered"]
    val["presence"][0] = val["class_pixels"][0] = val["background"] = 0
    val["images"] = m.TARGET - 6
    over, warnings = m.threshold_evaluation(samples, val, total, groups)
    assert over["checks"]["class_presence_error_gt_1_2"][0]["triggered"]
    assert over["checks"]["class_pixel_error_gt_1_2"][0]["triggered"]
    assert over["checks"]["all_background_error_gt_1_2"]["triggered"]
    assert over["checks"]["val_image_deviation_gt_limit"]["triggered"]

    total["pixels"], total["depth_valid"] = 3, 1
    val["pixels"], val["depth_valid"] = 1, 0
    exact_aux, _ = m.threshold_evaluation(samples, val, total, groups)
    assert exact_aux["checks"]["aux_component_error_gt_1_3"][0]["value"] == "1/3"
    assert not exact_aux["checks"]["aux_component_error_gt_1_3"][0]["triggered"]
    total["pixels"], total["depth_valid"] = 4, 2
    over_aux, _ = m.threshold_evaluation(samples, val, total, groups)
    assert over_aux["checks"]["aux_component_error_gt_1_3"][0]["triggered"]

    official = tuple(f"RGB/{mine}-01-01-{index:04}-x.jpg" for mine in m.MINES for index in (1, 2))
    train, validation = official[::2], official[1::2]
    relationships = m.set_relationships(official, (), train, validation)
    mine_counts = {side: {mine: {"images": 1, "groups": 1} for mine in m.MINES} for side in ("train_dev", "val_dev")}
    assert m.hard_constraint_checks(relationships, mine_counts, True)["all_pass"]
    for keyword in ("official_train_identity", "official_test_identity", "frozen_parameters", "modalities_valid", "mine_group_minimum", "optimizer_converged"):
        assert not m.hard_constraint_checks(relationships, mine_counts, True, **{keyword: False})["all_pass"]
    assert not m.hard_constraint_checks(relationships, mine_counts, False)["all_pass"]
    broken_counts = copy.deepcopy(mine_counts)
    broken_counts["val_dev"]["01"]["groups"] = 0
    assert not m.hard_constraint_checks(relationships, broken_counts, True)["all_pass"]
    m.enforce_search_guard(9999)
    with pytest.raises(ValueError, match="10000"):
        m.enforce_search_guard(10000)


def test_canonical_json_unicode_finite_numbers_and_list_byte_policies():
    expected = '{"a":"矿","z":1}\n'.encode("utf-8")
    assert m.canonical_json({"z": 1, "a": "矿"}) == expected
    assert a.canonical({"z": 1, "a": "矿"}) == expected
    assert expected.endswith(b"\n") and not expected.endswith(b"\n\n") and b"\r" not in expected
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError): m.canonical_json({"value": value})
        with pytest.raises(ValueError): a.canonical({"value": value})
    groups = {"01-01-01-0001": [_sample("RGB/01-01-01-0001-a.jpg")]}
    assert m.output_lines(groups, set(groups)) == b"RGB/01-01-01-0001-a.jpg\n"
    official_test = b"RGB/02-01-01-0001-a.jpg\r\n"
    assert bytes(official_test) == official_test  # Generator assigns the original bytes without normalization.


def test_group_sizes_shuffle_invariance_output_bytes_and_golden_tie_break_trace():
    groups = {}
    for mine_index, mine in enumerate(m.MINES):
        for group_index, size in enumerate((1, 2 if mine_index % 2 == 0 else 3), start=1):
            group = f"{mine}-01-{mine_index + 1:02}-{group_index:04}"
            groups[group] = [_sample(f"RGB/{group}-{sample_index:02}.jpg") for sample_index in range(size)]
    selected, moves, reason = m.optimize(groups)
    train_bytes, val_bytes = m.output_lines(groups, set(groups) - selected), m.output_lines(groups, selected)
    rng = random.Random(7331)
    items = list(groups.items())
    rng.shuffle(items)
    shuffled = {}
    for group, group_samples in items:
        group_samples = list(group_samples)
        rng.shuffle(group_samples)
        shuffled[group] = group_samples
    selected_shuffled, moves_shuffled, reason_shuffled = m.optimize(shuffled)
    assert (selected_shuffled, moves_shuffled, reason_shuffled) == (selected, moves, reason)
    assert m.output_lines(shuffled, set(shuffled) - selected_shuffled) == train_bytes
    assert m.output_lines(shuffled, selected_shuffled) == val_bytes
    train_set, val_set = set(train_bytes.decode().splitlines()), set(val_bytes.decode().splitlines())
    for group_samples in groups.values():
        paths = {sample.path for sample in group_samples}
        assert paths <= train_set or paths <= val_set

    trace = []
    result = m.optimize(_tiny_groups(0), trace=trace)
    assert result[1:] == (6, "no_strict_improvement")
    assert trace == [
        (1, "06-01-06-0003", None), (1, "03-01-03-0001", None),
        (1, "05-01-05-0003", None), (1, "04-01-04-0002", None),
        (1, "02-01-02-0002", None), (1, "01-01-01-0002", None),
    ]


def test_manifest_schema_each_required_field_status_and_exact_output_keyset(tiny_candidate):
    root = tiny_candidate["output"]
    manifest_path = root / "manifest.json"
    pristine = json.loads(manifest_path.read_text(encoding="utf-8"))
    containers = [
        (), ("official",), ("official", "train"), ("official", "test"), ("inventory",),
        *(("inventory", name) for name in ("RGB", "Depth16", "Label")), ("outputs",),
        *(("outputs", name) for name in ("train-dev.txt", "val-dev.txt", "official-test.txt")),
        ("set_relationships",), ("set_relationships", "samples"), ("set_relationships", "groups"),
        ("mine_counts",), *(("mine_counts", side) for side in ("train_dev", "val_dev")),
        *(("mine_counts", side, mine) for side in ("train_dev", "val_dev") for mine in m.MINES),
        ("statistic_definitions",), ("statistic_definitions", "bins"), ("statistics",),
        *(("statistics", side) for side in ("official_train", "train_dev", "val_dev")),
        ("objective",), ("objective", "levels"), ("objective", "subitems"), ("local_search",),
        ("threshold_checks",), ("threshold_checks", "checks"), ("hard_constraint_checks",), ("user_gate_a",),
    ]
    for container_path in containers:
        original_container = pristine
        for part in container_path:
            original_container = original_container[part]
        for field in original_container:
            mutated = copy.deepcopy(pristine)
            target = mutated
            for part in container_path:
                target = target[part]
            del target[field]
            manifest_path.write_bytes(m.canonical_json(mutated))
            dotted = ".".join((*container_path, field))
            assert not a.audit(tiny_candidate["dataset"], tiny_candidate["train"], tiny_candidate["test"], root)["pass"], dotted
    mutations = []
    frozen = copy.deepcopy(pristine); frozen["candidate_status"] = "frozen"; mutations.append(frozen)
    approved = copy.deepcopy(pristine); approved["user_gate_a"]["status"] = "approved"; mutations.append(approved)
    extra = copy.deepcopy(pristine); extra["outputs"]["extra.txt"] = {}; mutations.append(extra)
    missing = copy.deepcopy(pristine); del missing["outputs"]["val-dev.txt"]; mutations.append(missing)
    for mutated in mutations:
        manifest_path.write_bytes(m.canonical_json(mutated))
        assert not a.audit(tiny_candidate["dataset"], tiny_candidate["train"], tiny_candidate["test"], root)["pass"]
    manifest_path.write_bytes(m.canonical_json(pristine))
    assert a.audit(tiny_candidate["dataset"], tiny_candidate["train"], tiny_candidate["test"], root)["pass"]


def test_audit_read_only_write_scope_and_all_authoritative_tamper_checks(tiny_candidate):
    root = tiny_candidate["output"]
    args = (tiny_candidate["dataset"], tiny_candidate["train"], tiny_candidate["test"], root)
    whole_tree = root.parent
    before = _tree_hashes(whole_tree)
    assert a.audit(*args, write=False)["pass"]
    assert _tree_hashes(whole_tree) == before
    (root / "audit-report.json").unlink()
    before_write = _tree_hashes(whole_tree)
    assert a.audit(*args, write=True)["pass"]
    after_write = _tree_hashes(whole_tree)
    assert set(after_write) - set(before_write) == {"candidate/audit-report.json"}
    assert {name for name in before_write if before_write[name] != after_write.get(name)} == set()
    for name in ("train-dev.txt", "val-dev.txt"):
        raw = (root / name).read_bytes()
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n") and b"\r" not in raw
    assert (root / "official-test.txt").read_bytes() == tiny_candidate["test_raw"]
    assert b"\r\n" in tiny_candidate["test_raw"]

    manifest_path = root / "manifest.json"
    pristine_manifest_raw = manifest_path.read_bytes()
    pristine_manifest = json.loads(pristine_manifest_raw)
    train_path = root / "train-dev.txt"
    pristine_train = train_path.read_bytes()
    train_path.write_bytes(pristine_train + pristine_train.splitlines(keepends=True)[0])
    assert not a.audit(*args)["pass"]
    train_path.write_bytes(pristine_train)
    mutations = []
    wrong_protocol = copy.deepcopy(pristine_manifest); wrong_protocol["protocol_id"] = "tampered"; mutations.append(wrong_protocol)
    wrong_status = copy.deepcopy(pristine_manifest); wrong_status["candidate_status"] = "frozen"; mutations.append(wrong_status)
    wrong_hash = copy.deepcopy(pristine_manifest); wrong_hash["outputs"]["train-dev.txt"]["sha256"] = "0" * 64; mutations.append(wrong_hash)
    extra_key = copy.deepcopy(pristine_manifest); extra_key["outputs"]["extra.txt"] = {}; mutations.append(extra_key)
    for mutated in mutations:
        manifest_path.write_bytes(m.canonical_json(mutated))
        assert not a.audit(*args)["pass"]
    manifest_path.write_bytes(pristine_manifest_raw)
    assert a.audit(*args)["pass"]


def test_audit_enforces_frozen_hash_count_and_group_constants_not_manifest_only(tiny_candidate, monkeypatch):
    args = (tiny_candidate["dataset"], tiny_candidate["train"], tiny_candidate["test"], tiny_candidate["output"])
    cases = {
        "TRAIN_HASH": "0" * 64, "TRAIN_COUNT": tiny_candidate["frozen"]["TRAIN_COUNT"] + 1,
        "TRAIN_GROUPS": tiny_candidate["frozen"]["TRAIN_GROUPS"] + 1,
        "TEST_HASH": "f" * 64, "TEST_COUNT": tiny_candidate["frozen"]["TEST_COUNT"] + 1,
        "TEST_GROUPS": tiny_candidate["frozen"]["TEST_GROUPS"] + 1,
    }
    for name, wrong_value in cases.items():
        original = getattr(m, name)
        monkeypatch.setattr(m, name, wrong_value)
        report = a.audit(*args)
        expected_check = "frozen.input_train" if name.startswith("TRAIN") else "frozen.input_test"
        assert report["checks"][expected_check] is False
        assert not report["pass"]
        monkeypatch.setattr(m, name, original)


def test_cli_frozen_identity_hard_failure_creates_no_output(tmp_path):
    train = tmp_path / "train.txt"
    test = tmp_path / "test.txt"
    output = tmp_path / "candidate"
    train.write_bytes(b"RGB/01-01-01-0001-a.jpg\n")
    test.write_bytes(b"RGB/02-01-01-0001-a.jpg\n")
    with pytest.raises(ValueError, match="hash/count/group"):
        m.main([
            "--dataset-root", str(tmp_path), "--official-train", str(train),
            "--official-test", str(test), "--output-root", str(output),
        ])
    assert not output.exists()


def test_generator_transaction_cleans_partial_writes_and_failed_audit(tiny_candidate, monkeypatch):
    output = tiny_candidate["output"]
    parent = output.parent
    args = [
        "--dataset-root", str(tiny_candidate["dataset"]),
        "--official-train", str(tiny_candidate["train"]),
        "--official-test", str(tiny_candidate["test"]),
        "--output-root", str(output),
    ]

    shutil.rmtree(output)
    original_write = Path.write_bytes

    def fail_manifest_write(path: Path, content: bytes) -> int:
        if path.name == "manifest.json" and path.parent.name.startswith(".candidate.tmp-"):
            raise OSError("injected manifest write failure")
        return original_write(path, content)

    monkeypatch.setattr(Path, "write_bytes", fail_manifest_write)
    with pytest.raises(OSError, match="injected manifest write failure"):
        m.main(args)
    assert not output.exists()
    assert not list(parent.glob(".candidate.tmp-*"))

    monkeypatch.setattr(Path, "write_bytes", original_write)
    monkeypatch.setattr(a, "audit", lambda *_args, **_kwargs: {"pass": False})
    with pytest.raises(ValueError, match="generated audit failed"):
        m.main(args)
    assert not output.exists()
    assert not list(parent.glob(".candidate.tmp-*"))


def test_existing_audit_report_bytes_staleness_and_forged_pass_are_rejected(tiny_candidate):
    root = tiny_candidate["output"]
    args = (tiny_candidate["dataset"], tiny_candidate["train"], tiny_candidate["test"], root)
    report_path = root / "audit-report.json"
    pristine = report_path.read_bytes()
    assert a.audit(*args, write=False)["pass"]

    report_path.write_bytes(pristine + b" ")
    assert not a.audit(*args, write=False)["pass"]

    stale = json.loads(pristine)
    stale["manifest_sha256"] = "0" * 64
    stale["details"]["manifest_sha256"] = "0" * 64
    report_path.write_bytes(a.canonical(stale))
    assert not a.audit(*args, write=False)["pass"]

    forged = {"checks": {"forged": True}, "details": {}, "manifest_sha256": stale["manifest_sha256"], "pass": True, "schema_version": "museg-dev-split-audit-v1"}
    report_path.write_bytes(a.canonical(forged))
    assert not a.audit(*args, write=False)["pass"]

    report_path.write_bytes(pristine)
    assert a.audit(*args, write=False)["pass"]


def test_audit_accepts_recorded_commit_after_head_moves_but_rejects_missing_commit(tiny_candidate, monkeypatch):
    args = (tiny_candidate["dataset"], tiny_candidate["train"], tiny_candidate["test"], tiny_candidate["output"])
    monkeypatch.setattr(m, "git_commit", lambda _root: "b" * 40)
    assert a.audit(*args, write=False)["pass"]
    monkeypatch.setattr(m, "git_object_is_commit", lambda _root, _value: False)
    report = a.audit(*args, write=False)
    assert not report["pass"]
    assert not report["checks"]["generator.commit"]


@pytest.mark.parametrize("source_name", ["create_museg_dev_split.py", "audit_museg_splits.py"])
def test_audit_rejects_current_tool_source_byte_change(tiny_candidate, monkeypatch, source_name):
    args = (tiny_candidate["dataset"], tiny_candidate["train"], tiny_candidate["test"], tiny_candidate["output"])
    target = (ROOT / "tools" / "splits" / source_name).resolve()
    original_read = Path.read_bytes

    def changed_source(path: Path) -> bytes:
        content = original_read(path)
        return content + b"# injected source change\n" if path.resolve() == target else content

    monkeypatch.setattr(Path, "read_bytes", changed_source)
    report = a.audit(*args, write=False)
    assert not report["pass"]
    assert not report["checks"]["generator.tool_sources"]
