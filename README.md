# <img src="https://img.icons8.com/color/48/000000/artificial-intelligence.png" width="30"/> CoVE: Official Repository

[![ACL 2025](https://img.shields.io/badge/ACL-2025-blue)](https://acl2025.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Stars](https://img.shields.io/github/stars/HaochenZhang717/CoVE-official-Repo?style=social)](https://github.com/HaochenZhang717/CoVE-official-Repo)

**Official implementation** of our ACL 2025 paper:  
**"CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems"**

---

## <img src="https://img.icons8.com/color/48/000000/overview-pages-1.png" width="30"/> Overview

![Methodology Figure](https://github.com/HaochenZhang717/CoVE-official-Repo/raw/main/figure-1.png)

*Figure 1: Architecture overview of CoVE framework*

---

## <img src="https://img.icons8.com/color/48/000000/statistics.png" width="30"/> Compression Rate Analysis

![Compression Rate Performance](https://github.com/HaochenZhang717/CoVE-official-Repo/raw/main/compression_ratios.png)

*Figure 2: Performance across different compression rates (2-64)*

Key observations:
- ▶️ **Best performance** at compression rate = 2
- ↘️ Gradual performance degradation with higher compression
- ⚖️ Good trade-off between efficiency and accuracy at rate = 8

---

## <img src="https://img.icons8.com/color/48/000000/leaderboard.png" width="30"/> Performance Comparison

<div align="center" style="overflow-x: auto;">

| Dataset  | Metric | GRU4Rec | BERT4Rec | Caser | SASRec | P5 | HGN | S³-Rec | FDSA | TIGER | <span style="color:#2ecc71;font-weight:bold">CoVE</span> |
|----------|--------|---------|----------|-------|--------|----|-----|--------|------|-------|-------|
| **Beauty** | NG5 | 0.0099 | 0.0124 | 0.0131 | 0.0249 | 0.0107 | 0.0206 | 0.0244 | 0.0163 | <u>0.0321</u> | <span style="color:#2ecc71">**0.0498**</span> |
|          | NG10 | 0.0137 | 0.0170 | 0.0176 | 0.0318 | 0.0136 | 0.0266 | 0.0327 | 0.0208 | <u>0.0384</u> | <span style="color:#2ecc71">**0.0593**</span> |
|          | HR5 | 0.0164 | 0.0203 | 0.0205 | 0.0387 | 0.0163 | 0.0325 | 0.0387 | 0.0267 | <u>0.0454</u> | <span style="color:#2ecc71">**0.0714**</span> |
|          | HR10 | 0.0283 | 0.0347 | 0.0347 | 0.0605 | 0.0254 | 0.0512 | 0.0647 | 0.0407 | <u>0.0648</u> | <span style="color:#2ecc71">**0.1009**</span> |
| **Toys** | NG5 | 0.0059 | 0.0071 | 0.0107 | 0.0306 | 0.0050 | 0.0221 | 0.0294 | 0.0140 | <u>0.0371</u> | <span style="color:#2ecc71">**0.0509**</span> |
|          | NG10 | 0.0084 | 0.0099 | 0.0141 | 0.0374 | 0.0066 | 0.0277 | 0.0376 | 0.0189 | <u>0.0432</u> | <span style="color:#2ecc71">**0.0595**</span> |
|          | HR5 | 0.0097 | 0.0116 | 0.0166 | 0.0463 | 0.0070 | 0.0321 | 0.0443 | 0.0228 | <u>0.0521</u> | <span style="color:#2ecc71">**0.0719**</span> |
|          | HR10 | 0.0176 | 0.0203 | 0.0270 | 0.0675 | 0.0121 | 0.0497 | 0.0700 | 0.0381 | <u>0.0712</u> | <span style="color:#2ecc71">**0.0986**</span> |
| **Sports** | NG5 | 0.0086 | 0.0075 | 0.0072 | 0.0192 | 0.0041 | 0.0120 | <u>0.0204</u> | 0.0156 | 0.0181 | <span style="color:#2ecc71">**0.0296**</span> |
|            | NG10 | 0.0110 | 0.0099 | 0.0097 | <u>0.0249</u> | 0.0052 | 0.0159 | 0.0240 | 0.0156 | 0.0225 | <span style="color:#2ecc71">**0.0359**</span> |
|            | HR5 | 0.0129 | 0.0115 | 0.0116 | 0.0233 | 0.0061 | 0.0189 | 0.0251 | 0.0182 | <u>0.0264</u> | <span style="color:#2ecc71">**0.0428**</span> |
|            | HR10 | 0.0204 | 0.0191 | 0.0194 | 0.0350 | 0.0095 | 0.0313 | 0.0385 | 0.0288 | <u>0.0400</u> | <span style="color:#2ecc71">**0.0624**</span> |

</div>

*Table 1: Comparative results across three benchmark datasets.  
<span style="color:#2ecc71">**Green bold**</span> = CoVE results (compression rate=2), <u>Underlined</u> = Best baseline performance*

Key findings:
- 🏆 CoVE **outperforms all baselines** across all metrics
- 📈 Consistent improvements of **20-50%** over best baselines
- 💡 Particularly strong on **HR@10** metrics

---

## <img src="https://img.icons8.com/color/48/000000/download--v1.png" width="30"/> Dataset Download

Due to dataset size limitations, please download from:

🔗 [Google Drive Folder](https://drive.google.com/drive/folders/1h_swkdw4Evp7X4iNaOYczlf_vvCd_Px6?usp=share_link)

📦 File structure:
