#!/bin/bash
cd /home/lzy/project/slot-datamaking/model/research/papers

download() {
    local url="$1"
    local dest="$2"
    if [ ! -f "$dest" ]; then
        curl -sL --connect-timeout 10 --max-time 60 -o "$dest" "$url" 2>/dev/null
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

download "https://arxiv.org/pdf/2103.09504.pdf" "002_predrnn/predrnn.pdf" &
download "https://arxiv.org/pdf/2211.12509.pdf" "003_simvp/simvp.pdf" &
download "https://arxiv.org/pdf/1803.01814.pdf" "004_svg/svg.pdf" &
wait

download "https://arxiv.org/pdf/2002.10562.pdf" "005_phyre/phyre.pdf" &
download "https://arxiv.org/pdf/2011.13045.pdf" "006_physion/physion.pdf" &
download "https://arxiv.org/pdf/2203.09333.pdf" "007_kubric/kubric.pdf" &
download "https://arxiv.org/pdf/2006.15055.pdf" "008_slot_attention/slot_attention.pdf" &
wait

download "https://arxiv.org/pdf/2203.10147.pdf" "009_savi/savi.pdf" &
download "https://arxiv.org/pdf/2301.06007.pdf" "010_slotformer/slotformer.pdf" &
download "https://arxiv.org/pdf/1901.11370.pdf" "011_monet/monet.pdf" &
download "https://arxiv.org/pdf/1906.10963.pdf" "012_iodine/iodine.pdf" &
wait

download "https://arxiv.org/pdf/1907.13052.pdf" "013_genesis/genesis.pdf" &
download "https://arxiv.org/pdf/1612.00222.pdf" "014_interaction_networks/in.pdf" &
download "https://arxiv.org/pdf/1806.01261.pdf" "015_graph_networks/gn.pdf" &
download "https://arxiv.org/pdf/1710.10903.pdf" "016_gat/gat.pdf" &
wait

download "https://arxiv.org/pdf/1511.05493.pdf" "017_ggnn/ggnn.pdf" &
download "https://arxiv.org/pdf/1704.01212.pdf" "018_mpnn/mpnn.pdf" &
download "https://arxiv.org/pdf/1906.01737.pdf" "019_hamiltonian_nn/hnn.pdf" &
download "https://arxiv.org/pdf/2003.04630.pdf" "020_lagrangian_nn/lnn.pdf" &
wait

download "https://arxiv.org/pdf/2010.11929.pdf" "021_vit/vit.pdf" &
download "https://arxiv.org/pdf/2012.12877.pdf" "022_deit/deit.pdf" &
download "https://arxiv.org/pdf/2103.14030.pdf" "023_swin/swin.pdf" &
download "https://arxiv.org/pdf/2104.14294.pdf" "024_dino/dino.pdf" &
wait

download "https://arxiv.org/pdf/2304.07193.pdf" "025_dinov2/dinov2.pdf" &
download "https://arxiv.org/pdf/2111.06377.pdf" "026_mae/mae.pdf" &
download "https://arxiv.org/pdf/1506.04214.pdf" "027_convlstm/convlstm.pdf" &
download "https://arxiv.org/pdf/1811.09304.pdf" "028_e3d_lstm/e3d_lstm.pdf" &
wait

download "https://arxiv.org/pdf/2102.05095.pdf" "029_timesformer/timesformer.pdf" &
download "https://arxiv.org/pdf/2112.07109.pdf" "030_video_swin/video_swin.pdf" &
download "https://arxiv.org/pdf/2110.13985.pdf" "031_s4/s4.pdf" &
download "https://arxiv.org/pdf/2312.00752.pdf" "032_mamba/mamba.pdf" &
wait

download "https://arxiv.org/pdf/2107.03430.pdf" "033_ssim/ssim.pdf" &
download "https://arxiv.org/pdf/1801.03924.pdf" "034_lpips/lpips.pdf" &
download "https://arxiv.org/pdf/1812.04365.pdf" "035_fvd/fvd.pdf" &
download "https://arxiv.org/pdf/1708.02002.pdf" "036_focal_loss/focal.pdf" &
wait

download "https://arxiv.org/pdf/1606.04797.pdf" "037_dice_loss/dice.pdf" &
download "https://arxiv.org/pdf/1902.09630.pdf" "038_giou/giou.pdf" &
download "https://arxiv.org/pdf/2103.00020.pdf" "039_clip/clip.pdf" &
download "https://arxiv.org/pdf/2005.12872.pdf" "040_detr/detr.pdf" &
wait

download "https://arxiv.org/pdf/1911.05722.pdf" "041_moco/moco.pdf" &
download "https://arxiv.org/pdf/2205.14135.pdf" "042_flash_attention/flash_attn.pdf" &
download "https://arxiv.org/pdf/2009.14794.pdf" "043_performer/performer.pdf" &
download "https://arxiv.org/pdf/2006.04768.pdf" "044_linformer/linformer.pdf" &
wait

download "https://arxiv.org/pdf/1604.06174.pdf" "045_gradient_checkpointing/gc.pdf" &
download "https://arxiv.org/pdf/1710.03740.pdf" "046_amp/amp.pdf" &
wait

echo "=== ALL DOWNLOADS COMPLETE ==="
