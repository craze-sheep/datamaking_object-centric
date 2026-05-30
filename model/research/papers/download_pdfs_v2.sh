#!/bin/bash
cd /home/lzy/project/slot-datamaking/model/research/papers

download() {
    local url="$1"
    local dest="$2"
    if [ ! -f "$dest" ]; then
        curl -sL --connect-timeout 15 --max-time 120 -o "$dest" "$url" 2>/dev/null
        if [ $? -eq 0 ] && [ -s "$dest" ] && [ $(stat -c%s "$dest") -gt 10000 ]; then
            echo "OK: $dest ($(du -h $dest | cut -f1))"
        else
            rm -f "$dest"
            echo "FAIL: $dest"
        fi
    else
        echo "EXISTS: $dest"
    fi
}

# Use correct arxiv PDF URLs
echo "=== PDF Batch 1 ==="
download "https://arxiv.org/pdf/2003.01460v2.pdf" "001_phydnet/phydnet.pdf" &
download "https://arxiv.org/pdf/2103.09504v2.pdf" "002_predrnn/predrnn.pdf" &
download "https://arxiv.org/pdf/2206.05099v3.pdf" "003_simvp/simvp.pdf" &
download "https://arxiv.org/pdf/1802.07687v3.pdf" "004_svg/svg.pdf" &
wait

echo "=== PDF Batch 2 ==="
download "https://arxiv.org/pdf/1908.05656v2.pdf" "005_phyre/phyre.pdf" &
download "https://arxiv.org/pdf/2011.13045v2.pdf" "006_physion/physion.pdf" &
download "https://arxiv.org/pdf/2203.09333v1.pdf" "007_kubric/kubric.pdf" &
download "https://arxiv.org/pdf/2006.15055v2.pdf" "008_slot_attention/slot_attention.pdf" &
wait

echo "=== PDF Batch 3 ==="
download "https://arxiv.org/pdf/2203.10147v2.pdf" "009_savi/savi.pdf" &
download "https://arxiv.org/pdf/2302.06109v2.pdf" "010_slotformer/slotformer.pdf" &
download "https://arxiv.org/pdf/1901.11370v2.pdf" "011_monet/monet.pdf" &
download "https://arxiv.org/pdf/1906.10963v2.pdf" "012_iodine/iodine.pdf" &
wait

echo "=== PDF Batch 4 ==="
download "https://arxiv.org/pdf/1907.13052v2.pdf" "013_genesis/genesis.pdf" &
download "https://arxiv.org/pdf/1612.00222v1.pdf" "014_interaction_networks/in.pdf" &
download "https://arxiv.org/pdf/1806.01261v4.pdf" "015_graph_networks/gn.pdf" &
download "https://arxiv.org/pdf/1710.10903v3.pdf" "016_gat/gat.pdf" &
wait

echo "=== PDF Batch 5 ==="
download "https://arxiv.org/pdf/1511.05493v2.pdf" "017_ggnn/ggnn.pdf" &
download "https://arxiv.org/pdf/1704.01212v2.pdf" "018_mpnn/mpnn.pdf" &
download "https://arxiv.org/pdf/1906.01737v2.pdf" "019_hamiltonian_nn/hnn.pdf" &
download "https://arxiv.org/pdf/2003.04630v2.pdf" "020_lagrangian_nn/lnn.pdf" &
wait

echo "=== PDF Batch 6 ==="
download "https://arxiv.org/pdf/2010.11929v2.pdf" "021_vit/vit.pdf" &
download "https://arxiv.org/pdf/2012.12877v3.pdf" "022_deit/deit.pdf" &
download "https://arxiv.org/pdf/2103.14030v2.pdf" "023_swin/swin.pdf" &
download "https://arxiv.org/pdf/2104.14294v3.pdf" "024_dino/dino.pdf" &
wait

echo "=== PDF Batch 7 ==="
download "https://arxiv.org/pdf/2304.07193v2.pdf" "025_dinov2/dinov2.pdf" &
download "https://arxiv.org/pdf/2111.06377v2.pdf" "026_mae/mae.pdf" &
download "https://arxiv.org/pdf/1506.04214v5.pdf" "027_convlstm/convlstm.pdf" &
download "https://arxiv.org/pdf/1811.09304v1.pdf" "028_e3d_lstm/e3d_lstm.pdf" &
wait

echo "=== PDF Batch 8 ==="
download "https://arxiv.org/pdf/2102.05095v2.pdf" "029_timesformer/timesformer.pdf" &
download "https://arxiv.org/pdf/2106.13230v3.pdf" "030_video_swin/video_swin.pdf" &
download "https://arxiv.org/pdf/2110.13985v2.pdf" "031_s4/s4.pdf" &
download "https://arxiv.org/pdf/2312.00752v2.pdf" "032_mamba/mamba.pdf" &
wait

echo "=== PDF Batch 9 ==="
download "https://arxiv.org/pdf/1801.03924v3.pdf" "034_lpips/lpips.pdf" &
download "https://arxiv.org/pdf/1812.04365v2.pdf" "035_fvd/fvd.pdf" &
download "https://arxiv.org/pdf/1708.02002v3.pdf" "036_focal_loss/focal.pdf" &
download "https://arxiv.org/pdf/1606.04797v1.pdf" "037_dice_loss/dice.pdf" &
wait

echo "=== PDF Batch 10 ==="
download "https://arxiv.org/pdf/1902.09630v2.pdf" "038_giou/giou.pdf" &
download "https://arxiv.org/pdf/2103.00020v1.pdf" "039_clip/clip.pdf" &
download "https://arxiv.org/pdf/2005.12872v3.pdf" "040_detr/detr.pdf" &
download "https://arxiv.org/pdf/1911.05722v3.pdf" "041_moco/moco.pdf" &
wait

echo "=== PDF Batch 11 ==="
download "https://arxiv.org/pdf/2205.14135v2.pdf" "042_flash_attention/flash_attn.pdf" &
download "https://arxiv.org/pdf/2009.14794v2.pdf" "043_performer/performer.pdf" &
download "https://arxiv.org/pdf/2006.04768v1.pdf" "044_linformer/linformer.pdf" &
download "https://arxiv.org/pdf/1710.03740v3.pdf" "046_amp/amp.pdf" &
wait

echo "=== PDF Batch 12 ==="
download "https://arxiv.org/pdf/1705.07115v4.pdf" "050_multitask_loss/mtl.pdf" &
download "https://arxiv.org/pdf/0912.4940v4.pdf" "049_curriculum_learning/curriculum.pdf" &
download "https://arxiv.org/pdf/2006.15055v2.pdf" "048_ocvp/ocvp.pdf" &
wait

echo "=== ALL PDF DOWNLOADS DONE ==="
