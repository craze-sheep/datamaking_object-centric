#!/bin/bash
cd /home/lzy/project/slot-datamaking/model/research/papers

clone() {
    local url="$1"
    local dest="$2"
    if [ ! -d "$dest" ] || [ -z "$(ls -A $dest 2>/dev/null)" ]; then
        rm -rf "$dest"
        git clone --depth 1 "$url" "$dest" 2>/dev/null
        if [ $? -eq 0 ] && [ -n "$(ls -A $dest 2>/dev/null)" ]; then
            echo "OK: $dest"
        else
            rm -rf "$dest"
            echo "FAIL: $dest"
        fi
    else
        echo "EXISTS: $dest"
    fi
}

download() {
    local url="$1"
    local dest="$2"
    if [ ! -f "$dest" ]; then
        curl -sL --connect-timeout 15 --max-time 120 -o "$dest" "$url" 2>/dev/null
        if [ $? -eq 0 ] && [ -s "$dest" ]; then
            echo "OK: $dest"
        else
            rm -f "$dest"
            echo "FAIL: $dest"
        fi
    else
        echo "EXISTS: $dest"
    fi
}

echo "=== Batch 1: Physics Video Prediction ==="
clone "https://github.com/vincent-leguen/PhyDNet.git" "001_phydnet/code" &
clone "https://github.com/thuml/predrnn-pytorch.git" "002_predrnn/code" &
clone "https://github.com/chengtan9907/SimVP.git" "003_simvp/code" &
wait

echo "=== Batch 2: Object-Centric ==="
clone "https://github.com/google-research/slot-attention.git" "008_slot_attention/code" &
clone "https://github.com/google-research/savi.git" "009_savi/code" &
clone "https://github.com/pitaalfred/slotformer.git" "010_slotformer/code" &
wait

echo "=== Batch 3: Object-Centric continued ==="
clone "https://github.com/deepmind/multi-object-networks.git" "011_monet/code" &
clone "https://github.com/deepmind/deepmind-research.git" "012_iodine/code" &
clone "https://github.com/applied-ai-lab/genesis.git" "013_genesis/code" &
wait

echo "=== Batch 4: GNN ==="
clone "https://github.com/deepmind/graph_nets.git" "015_graph_networks/code" &
clone "https://github.com/google-deepmind/interaction_network.git" "014_interaction_networks/code" &
clone "https://github.com/Diego999/GAT.git" "016_gat/code" &
wait

echo "=== Batch 5: Physics NN ==="
clone "https://github.com/greydanus/hamiltonian-nn.git" "019_hamiltonian_nn/code" &
clone "https://github.com/MilesCranmer/lagrangian_nns.git" "020_lagrangian_nn/code" &
wait

echo "=== Batch 6: Vision ==="
clone "https://github.com/facebookresearch/dino.git" "024_dino/code" &
clone "https://github.com/facebookresearch/dinov2.git" "025_dinov2/code" &
clone "https://github.com/facebookresearch/mae.git" "026_mae/code" &
wait

echo "=== Batch 7: Video Models ==="
clone "https://github.com/SwinTransformer/Swin-Transformer-Video.git" "030_video_swin/code" &
clone "https://github.com/state-spaces/mamba.git" "032_mamba/code" &
clone "https://github.com/HazyResearch/state-spaces.git" "031_s4/code" &
wait

echo "=== Batch 8: Loss & Detection ==="
clone "https://github.com/facebookresearch/detr.git" "040_detr/code" &
clone "https://github.com/facebookresearch/moco-v3.git" "041_moco/code" &
clone "https://github.com/Dao-AILab/flash-attention.git" "042_flash_attention/code" &
wait

echo "=== Batch 9: Benchmarks ==="
clone "https://github.com/facebookresearch/phyre.git" "005_phyre/code" &
clone "https://github.com/daniel-bear/physion.git" "006_physion/code" &
clone "https://github.com/google-research/kubric.git" "007_kubric/code" &
wait

echo "=== Batch 10: Efficient Attention ==="
clone "https://github.com/lucidrains/performer-pytorch.git" "043_performer/code" &
clone "https://github.com/lucidrains/linformer.git" "044_linformer/code" &
clone "https://github.com/openai/CLIP.git" "039_clip/code" &
wait

echo "=== ALL CLONES DONE ==="
