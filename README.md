# <img src="https://img.icons8.com/color/48/000000/artificial-intelligence.png" width="30"/> CoVE: Official Repository

[![ACL 2025](https://img.shields.io/badge/ACL-2025-blue)](https://acl2025.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Stars](https://img.shields.io/github/stars/HaochenZhang717/CoVE-official-Repo?style=social)](https://github.com/HaochenZhang717/CoVE-official-Repo)

**Official implementation** of our ACL 2025 paper:  
**"CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems"**

---

## 📄 Overview

![Methodology Figure](https://github.com/HaochenZhang717/CoVE-official-Repo/raw/main/figure-1.png)

*Figure 1: Overview of the CoVE framework*

CoVE is a lightweight yet powerful vocabulary compression technique that enhances large language model (LLM) recommendations by adapting the vocabulary space to the target domain efficiently.

---

## 📉 Compression Rate Analysis

![Compression Rate Performance](https://github.com/HaochenZhang717/CoVE-official-Repo/raw/main/compression_ratios.png)

*Figure 2: Model performance at varying compression rates (2–64)*

**Key observations:**
- 🔝 Best results observed at compression rate = 2
- ⬇️ Performance gradually drops with more aggressive compression
- ⚖️ Balanced trade-off between speed and performance at rate = 8

---

## 🏁 Performance Comparison

<div align="center" style="overflow-x: auto;">

<!-- Results Table -->

</div>

*Table 1: Comparison on benchmark datasets. Green = CoVE (rate=2), Underlined = best non-CoVE baseline.*

**Key Takeaways:**
- 🚀 CoVE consistently **outperforms** all baselines across datasets
- 📊 Achieves **20–50%** improvements over strong baselines

---

## 📥 Dataset Download

Due to size limitations, datasets can be accessed via:

🔗 [Google Drive Folder](https://drive.google.com/drive/folders/1h_swkdw4Evp7X4iNaOYczlf_vvCd_Px6?usp=share_link)

📁 **File structure**:
```
datasets/
├── beauty/
├── toys/
└── sports/
```

---

## 🧪 Reproducibility

1. **Set up environment**:
```bash
bash install_env.sh
```

2. **Run training script** (example below, customize as needed):
```bash
python run_cove.py --dataset beauty --compression_rate 2 --epochs 100
```

3. **Evaluation**:
```bash
python evaluate.py --dataset beauty --model_checkpoint checkpoints/beauty_cr2/
```

---

## 📚 Citation

If you find this repository useful, please consider citing our paper:

```
@inproceedings{zhang2025cove,
  title={CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems},
  author={Zhang, Haochen and Others, Placeholder},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2025},
  url={https://github.com/HaochenZhang717/CoVE-official-Repo}
}
```

---

## 📬 Contact

If you have any questions or suggestions, feel free to open an issue or contact us directly.

---

Thank you for your interest in **CoVE**!