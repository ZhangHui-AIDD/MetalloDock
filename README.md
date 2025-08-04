# 🧲 **MetalloDock**

**AI-powered molecular docking for metalloproteins**
<img width="416" height="214" alt="image" src="https://github.com/user-attachments/assets/36f7770c-722b-4c67-be7d-afb4b7a0589c" />

## 🧬 About
**MetalloDock** is an AI-driven molecular docking framework tailored for **metalloproteins**. By combining autoregressive spatial decoding with physics-constrained geometric generation, MetalloDock excels at:

🔹 Reconstructing **metal coordination geometries**
🔹 Capturing accurate **metal-ligand interactions**
🔹 Outperforming existing tools in docking & virtual screening benchmarks

💡 **Real-world highlights**:

* Identified novel hits for **prostate-specific membrane antigen (PSMA)**
* Guided inhibitor design for **acidic polymerase endonuclease (PA)**

## ⚙️ Key Features

* 🧲 Metalloprotein-ligand **molecular docking**
* 🔬 **Binding affinity** prediction for metalloprotein complexes
* 🧪 **Virtual screening** against metalloprotein targets
* 🧠 Ligand **coordination atom prediction**
* 📐 **Metal coordination geometry reconstruction**

> 💡 *Additional functionalities for metalloprotein drug design are in active development.*

## 📦 Installation

```bash
git clone https://github.com/XXX.git
cd MetalloDock
conda create -n MetalloDock_env python=3.9
conda activate MetalloDock_env
pip install -r requirement.txt
```

## 📁 Project Structure

```
MetalloDock/
├── trained_models/          # Pretrained model weights
├── model/                   # Core MetalloDock model
├── dataset/                 # Dataset files and loaders
├── utils/                   # Scripts and utility tools
└── README.md                # Documentation
```

## 🚀 Demos & Usage

### 🧪 Demo 1: Ligand Docking on Metalloproteins

> Example files: `~/MetalloDock/dataset/metallo_vs/`

#### 1️⃣ Prepare Metalloprotein Data

```bash
cd ~/MetalloDock/utils
python pre_process.py --complex_path ~/MetalloDock/dataset/metallo_complex --pdbid_list 7oyo
```

**⚠ Requirements**:

* PDB **must include metal coordination bonds**
* **Remove water/heteroatoms/extraneous metals**
* **Preserve ligand residue names**

Update: `~/MetalloDock/dataset/test_pdbid.csv`

#### 2️⃣ Generate Complex Graphs

```bash
cd ~/MetalloDock/utils
python generate_graph.py \
--complex_path ~/MetalloDock/dataset/metallo_complex \
--graph_file_dir ~/MetalloDock/dataset/test_graph \
--frag_voc_file ~/MetalloDock/dataset/chembl_fragments_voc_dict.pkl \
--graph_path ~/MetalloDock/dataset/test_graph \
--csv_file ~/MetalloDock/dataset/test_pdbid.csv
```

#### 3️⃣ Perform Ligand Docking

```bash
cd ~/MetalloDock/utils
python ligand_docking.py \
--graph_file_dir ~/MetalloDock/dataset/test_graph \
--out_dir ~/MetalloDock/utils/predicted_poses/docking_test \
--protein_path ~/MetalloDock/dataset/metallo_complex \
--csv_file ~/MetalloDock/dataset/test_pdbid.csv
```

### 📈 Demo 2: Binding Affinity Prediction

```bash
cd ~/MetalloDock/utils
python affinity_scoring.py \
--ligand_path ~/MetalloDock/utils/predicted_poses/docking_test/0 \
--ligand_name 7oyo_pred \
--pocket_file ~/MetalloDock/dataset/metallo_complex/7oyo/7oyo_pro_pocket.pdb \
--mode docking \
--out_dir ~/MetalloDock/utils
```

### 🧭 Demo 3: Docking with Custom Coordination Atoms

```bash
cd ~/MetalloDock/utils
python ligand_docking_custom.py \
--ligand_path ~/MetalloDock/dataset/metallo_complex/7oyo \
--ligand_name 7oyo_ligand \
--pocket_file ~/MetalloDock/dataset/metallo_complex/7oyo/7oyo_pro_pocket.pdb \
--coordination_idx 12 \
--out_dir ~/MetalloDock/utils/predicted_poses/docking_custom
```

### 🧬 Demo 4: Virtual Screening Against Metalloprotein Targets

> Example target: **acidic polymerase endonuclease (PA)**
> Example dir: `~/MetalloDock/dataset/metallo_vs/PA/`

```bash
cd ~/MetalloDock/utils
python virtual_screening.py \
--out_dir ~/MetalloDock/dataset/metallo_vs/PA/result \
--ligand_file ~/MetalloDock/dataset/metallo_vs/PA/ligands \
--pocket_file ~/MetalloDock/dataset/metallo_vs/PA/receptor_pocket.pdb
```

✅ Or simply:

```bash
cd ~/MetalloDock/utils
python virtual_screening.py --target PA
```

## 🤝 Citation

*Coming soon*


## 📬 Contact

Feel free to reach out for collaboration or questions!

📧 Email: `22319063@zju.edu.cn`
📍 Institution: `Zhejiang University`

---

是否还需要加入 logo 图、图片示例或 GIF 演示？如果你有这些资源，我可以帮你插入并排版成更专业的文档格式。
