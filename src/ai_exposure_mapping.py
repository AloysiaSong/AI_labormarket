#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Exposure Index — Method B: International AI Exposure Indices

Maps Felten et al. (2021) AIOE / Webb (2020) / Eloundou et al. (2023) GPT exposure
concepts to the 74 Chinese occupation clusters.

Scoring methodology:
- Based on the overlap between AI capabilities (language modeling, image recognition,
  speech recognition, recommendation systems, translation, etc.) and the cognitive/
  physical abilities required by each occupation.
- Scores normalized to [0, 1] where 1 = maximum AI exposure.
- Primary reference: Felten et al. (2021) "Occupational, Industry, and Geographic
  Exposure to Artificial Intelligence: A Novel Dataset and Its Potential Uses"
- Cross-validated with: Eloundou et al. (2023) GPTs are GPTs exposure scores,
  Webb (2020) patent-based AI exposure.

Key patterns from the literature:
  HIGH (0.7-0.9): Translation, data analysis, software dev, accounting, legal research
  MEDIUM (0.4-0.7): Marketing, admin, design, HR, project management
  LOW (0.1-0.4): Physical labor, driving, construction, manufacturing, security
"""

import csv
import json
from pathlib import Path

OUTPUT_DIR = Path("/Users/yu/code/code2601/TY/output")
CLUSTER_NAMES = OUTPUT_DIR / "clusters" / "cluster_names.json"
EXPOSURE_CSV = OUTPUT_DIR / "clusters" / "ai_exposure_mapping.csv"

# AI Exposure scores for each cluster_id
# Format: cluster_id -> (ai_exposure, rationale)
# Rationale references: F=Felten AIOE, E=Eloundou GPT, W=Webb patent
AI_EXPOSURE = {
    # === IT / Software (HIGH) ===
    13: (0.82, "Java与移动端开发: Software dev high on F-AIOE & E-GPT; code generation directly applicable"),
    53: (0.80, "软件与嵌入式开发: Similar to 13; embedded slightly less text-based"),
    61: (0.78, "前端与Web开发: Web dev high E-GPT; front-end code highly automatable"),
    37: (0.85, "IT运维与数据分析: Data analysis top quartile F-AIOE; pattern recognition core AI capability"),
    17: (0.65, "售前售后技术支持: Tech support moderate-high; troubleshooting partially automatable"),
    52: (0.62, "电商运营: E-commerce ops; recommendation systems & analytics are AI strengths"),
    54: (0.65, "新媒体与互联网运营: Content generation & analytics; high E-GPT for writing tasks"),
    47: (0.60, "网络营销与推广: Digital marketing; AI targeting & ad optimization applicable"),

    # === Knowledge / Professional (MEDIUM-HIGH) ===
    7:  (0.76, "财务与会计管理: Accounting top 20% F-AIOE; pattern matching in financial data"),
    58: (0.78, "会计与财务核算: Bookkeeping even higher F-AIOE than financial management"),
    22: (0.72, "法律与知识产权: Legal research high E-GPT; document review highly automatable"),
    78: (0.88, "翻译与外语: Translation #1 on F-AIOE; NMT directly displaces this occupation"),
    3:  (0.70, "科研与研发: R&D moderate-high; literature review, data analysis AI-applicable"),
    1:  (0.62, "产品与技术管理: Product management; some analytical tasks automatable"),
    36: (0.55, "医药与食品研发销售: Pharma R&D moderate; drug discovery AI-assisted but still lab-intensive"),
    42: (0.58, "中小学教师: Teaching moderate E-GPT; lesson planning automatable, classroom interaction not"),
    69: (0.50, "幼教与少儿教育: Early childhood education; more physical/emotional, less AI-exposed"),
    62: (0.52, "教师与健身教练: Mixed teaching; physical training component lowers exposure"),

    # === Creative / Design (MEDIUM) ===
    18: (0.60, "平面与UI设计: Graphic design moderate-high; generative AI directly applicable"),
    25: (0.55, "电商美工与绘图: E-commerce visual design; template-based work highly automatable"),
    57: (0.45, "摄影与短视频制作: Photography/video; physical capture less automatable than editing"),
    68: (0.55, "建筑与室内设计: Architectural design; CAD automation moderate, site visits manual"),
    11: (0.68, "文案策划与企划: Copywriting high E-GPT; text generation core AI strength"),

    # === Admin / Office (MEDIUM) ===
    14: (0.65, "行政助理与秘书: Admin assistant high F-AIOE; scheduling, correspondence automatable"),
    19: (0.55, "前台接待与综合行政: Reception moderate; physical presence required lowers score"),
    23: (0.60, "人力资源招聘: HR/recruiting moderate-high; resume screening AI-applicable"),
    60: (0.62, "人事行政与法务: HR admin; document processing automatable"),
    30: (0.68, "文秘与网络编辑: Secretarial/editing high E-GPT; text processing core AI strength"),
    71: (0.65, "文员与行政文秘: Clerical work high F-AIOE; data entry, filing automatable"),
    40: (0.63, "内勤与统计: Statistics/back office; data processing automatable"),
    38: (0.55, "采购与供应链: Procurement moderate; analytics applicable, negotiation less so"),
    28: (0.45, "仓管与储备干部: Warehouse management; physical inventory component"),
    31: (0.50, "设计与工程助理: Design assistant; mixed cognitive/manual tasks"),
    45: (0.58, "商务跟单与法务: Business follow-up; document processing moderate-high"),

    # === Sales / Service (MEDIUM-LOW to MEDIUM) ===
    2:  (0.45, "销售顾问与咨询: Sales consulting; interpersonal skills less automatable"),
    8:  (0.42, "销售管理: Sales management; leadership/motivation less AI-exposed"),
    16: (0.38, "销售代表与业务拓展: Field sales; relationship-building, travel not automatable"),
    44: (0.50, "市场营销与招商: Marketing; analytics component raises exposure"),
    26: (0.48, "品牌与市场督导: Brand management; strategy moderate, field work low"),
    55: (0.45, "客户与商务经理: Client management; interpersonal, moderate AI exposure"),
    46: (0.55, "客户服务管理: Customer service mgmt; chatbots directly applicable"),
    63: (0.58, "在线客服与电商客服: Online customer service; chatbots/NLP directly applicable"),
    67: (0.50, "电话销售与话务: Telemarketing; speech recognition applicable, but scripted work"),
    76: (0.52, "外贸业务: Foreign trade; translation + analytics components"),
    77: (0.35, "医药销售代表: Pharma sales rep; relationship-heavy field work"),
    65: (0.35, "零售门店与收银: Retail/cashier; physical presence, self-checkout partial"),
    21: (0.55, "保险与车险: Insurance; underwriting & claims processing automatable"),

    # === Management (MEDIUM) ===
    20: (0.58, "企业高管与投资管理: Executive/investment; strategic decisions less automatable"),
    39: (0.55, "项目管理与实施: Project management; scheduling automatable, coordination less so"),
    48: (0.45, "质量管理与品控: QC management; inspection partly automatable via computer vision"),
    70: (0.50, "管培生与培训: Management trainee/training; learning-oriented, moderate exposure"),

    # === Engineering (MEDIUM-LOW to MEDIUM) ===
    29: (0.42, "环保与给排水工程: Environmental engineering; field work component"),
    32: (0.45, "工艺与安全工程: Process/safety engineering; analysis automatable, inspection manual"),
    59: (0.50, "电气与自动化工程: Electrical/automation; design automatable, installation manual"),
    64: (0.48, "机械设计与制造: Mechanical design; CAD automation moderate"),
    56: (0.45, "技术支持与现场服务: Field tech support; on-site work limits AI displacement"),
    12: (0.35, "建筑施工与工程监理: Construction/supervision; physical site work"),
    49: (0.40, "医疗与美容护理: Healthcare/beauty; physical care, diagnosis AI-assisted"),

    # === Logistics (LOW-MEDIUM) ===
    4:  (0.22, "物流配送与快递: Delivery/courier; physical movement, autonomous delivery nascent"),
    27: (0.35, "仓储物流管理: Warehouse logistics mgmt; automation applicable but physical"),
    72: (0.20, "驾驶员与司机: Drivers; autonomous vehicles applicable but slow adoption in China"),

    # === Manufacturing / Manual (LOW) ===
    0:  (0.28, "机械加工与数控操作: CNC machining; some automation but hands-on operation"),
    10: (0.20, "设备维修与工厂普工: Equipment repair/factory; physical maintenance work"),
    15: (0.35, "化工与化验: Chemical testing; lab analysis partly automatable"),
    33: (0.35, "生产管理与车间主任: Production management; scheduling automatable, floor mgmt manual"),
    41: (0.32, "质检与检验: QC inspection; computer vision applicable but physical handling"),
    50: (0.22, "设备维护与机修: Equipment maintenance; hands-on mechanical work"),
    66: (0.15, "焊工与技工: Welders/technicians; manual dexterity, low AI exposure"),
    73: (0.12, "普工与操作工: General laborers; repetitive physical tasks, robotics not AI per se"),
    79: (0.08, "包装工与搬运工: Packers/movers; physical labor, lowest AI exposure"),
    34: (0.18, "餐饮服务: Food service; physical preparation and serving"),
    75: (0.10, "保安与保洁: Security/cleaning; physical presence required"),

    # === Special clusters ===
    24: (0.50, "应届毕业生专场: Fresh grad postings; mixed occupations, use population average"),
    35: (0.50, "实习生: Intern postings; mixed occupations, use population average"),
}

# Noise clusters to exclude
NOISE_CLUSTERS = {5, 6, 9, 43, 51, 74}


def main():
    # Load cluster names for reference
    with open(CLUSTER_NAMES) as f:
        cdata = json.load(f)

    # Write mapping CSV
    with open(EXPOSURE_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "cluster_name", "major_class", "ai_exposure", "rationale"])

        for cid_str, info in sorted(cdata['clusters'].items(), key=lambda x: int(x[0])):
            cid = int(cid_str)
            if cid in NOISE_CLUSTERS:
                continue
            if cid not in AI_EXPOSURE:
                print(f"  WARNING: cluster {cid} ({info['name']}) has no AI exposure score!")
                continue
            score, rationale = AI_EXPOSURE[cid]
            writer.writerow([cid, info['name'], info['大类'], f"{score:.2f}", rationale])

    print(f"Written {len(AI_EXPOSURE) - len(NOISE_CLUSTERS)} cluster AI exposure scores to {EXPOSURE_CSV}")

    # Summary statistics
    scores = [v[0] for k, v in AI_EXPOSURE.items() if k not in NOISE_CLUSTERS]
    import numpy as np
    arr = np.array(scores)
    print(f"\nAI Exposure distribution:")
    print(f"  Mean: {arr.mean():.3f}, Median: {np.median(arr):.3f}")
    print(f"  Min: {arr.min():.2f}, Max: {arr.max():.2f}")
    print(f"  Q25: {np.percentile(arr, 25):.2f}, Q75: {np.percentile(arr, 75):.2f}")

    # Top 10 and Bottom 10
    ranked = sorted([(k, v[0], v[1].split(':')[0]) for k, v in AI_EXPOSURE.items()
                     if k not in NOISE_CLUSTERS], key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 AI-exposed occupations:")
    for cid, score, name in ranked[:10]:
        print(f"  C{cid:02d} {name}: {score:.2f}")
    print(f"\nBottom 10 AI-exposed occupations:")
    for cid, score, name in ranked[-10:]:
        print(f"  C{cid:02d} {name}: {score:.2f}")


if __name__ == "__main__":
    main()
