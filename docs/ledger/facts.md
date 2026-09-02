# Wilson Current 稳定事实台账

## 仓库与 ownership

- 活动本地 Git 工作区是 `C:/Users/Lenovo/Project/lattice-flow-restart`，remote `origin` 指向 `git@github.com:Si-Yang-Chen/EasyDistillation.git`。
- 当前分支、HEAD、remote refs 和 dirty 状态属于动态观察，记录于 `TASKBOARD.md` 的时间戳 snapshot；上线前重新 fetch/核验。
- `experiments/nonlocal-current/` 归本项目 owner；`experiments/localized-blending/` 归另一 owner。共享 `lattice/*` 改动需兼容集成，不得覆盖另一 agent。
- 与本项目路径、API、schema、数据格式和 runtime 均无交集的其他 agent 改动，在记录 commit 后原样接受，无需逐行审查；这不构成本项目对其正确性的背书。

## Current 实现

- `lattice.insertion.current` 定义版本化 Current terms、local/vector/axial/pseudoscalar operators、Wilson point-split `ConservedVectorCurrent`、端点解析、spin-aware assembly 和验证接口。
- Conserved vector current 每个方向有两个 terms：forward `-1/2(r-gamma_mu)` 和 backward `+1/2(r+gamma_mu)`；temporal component 使用不同 bar/field 时间。
- `lattice.insertion.gauge_link.DirectedCurrentBasis` 定义八方向顺序 `+x,+y,+z,-x,-y,-z,+t,-t`。
- `CurrentElementalGenerator.calc_directed_current_raw()` 保留四方向 gauge links，并生成轴为 `(direction,time,momentum,sink_ne,source_ne)` 的 temporal-capable V2V raw basis。
- legacy spatial `calc_all()`/`current_elemental_all` 仍是独立 schema，不能满足 temporal-current contract。
- `CurrentVertexAdapter` 仅消费 equal-time assembled V2V，明确拒绝 temporal point-split terms。

## Artifact 与 contraction

- `save_directed_current_v2v()`/`load_directed_current_v2v()` 使用内容寻址 NPY、原子 manifest、配置/动量/raw-contract、数据 hash 和 gauge/eigenvector source hashes。
- `contract_directed_current_v2v()` 逐 term 消费已加载 VSV：`S(field->sink) J(bar<-field) S(source->bar)`；backward raw channel 不二次 dagger。
- `contract_directed_current_pair_v2v()` 枚举两个 temporal currents 的四个 term pairs，以 `S(field_A->bar_B)` 和 `S(field_B->bar_A)` 闭合 ordered connected V2V trace。
- pair kernel 不隐含 Wick sign、flavor factor、体积 normalization、conjugation、real-part selection、source average 或 fit。

## 验证证据规则

- 测试数量、当前工作区状态和 reviewer verdict 是动态 evidence，记录于 `TASKBOARD.md`，不固化为稳定事实。
- deterministic synthetic precheck 与真实 smoke 必须继续分层；合成证据不等于真实物理结论。
- `tools/build_handover_manifest.py` 对 `docs/delivery-files.list` 的逐文件集合生成/验证 exact-byte SHA-256 manifest，并将文档验证与永久 release-ready 分成两个 gate。

## Kunshan 单流真实 smoke

- cfg10000 的已有 `Ne=1` VSV source-timeslice family 已审计；第二时间轴明确为 `source-relative`。
- directed-current generation accepted Job ID 为 `120571967`；Current artifact identity 为 `bb322b7d704cc3e7b551c12e22ca625de8300e529831fe94980b586f28ae280f`。
- temporal single-current smoke (`source=0,sink=8,current=4,J4,r=1`) result SHA-256 为 `f31f3975dbb4493d6eb0bb1c7bbcdf9ceaa7e2f30ab529d7048a3b6404880ac8`。
- 该结果是 real-artifact smoke，不是 Ward--Takahashi、charge normalization 或 ensemble physics result。

## Kunshan 8-cfg 输入

目标配置为 `10000,13000,14000,15000,16000,17000,18000,19000`。只读完整性核验确认：

- localized VSV：576/576 rank slabs；
- localized PSV：2304/2304 rank slabs；
- localized PSP：576/576 rank slabs；
- overlap matrices：8/8；
- gauge、eigenvectors、points 和 candidate C2 families 存在。

具体绝对路径、shape、dtype、axes、manifest hashes 和 compatibility verdict 见 `docs/kunshan-easydistillation-data-map.md`。

## 不兼容与 superseded 事实

- `03.current_elemental_all` 使用 legacy six-spatial-direction GaugeLink/`GammaName.A0` 定义，不是新 Wilson temporal `J4` artifact。
- `05.correlator.nocurrent.nodisp` 与 `05.correlator.nocurrent.nonlocal` 是不同 C2 families，bytes/hashes 不同。
- 历史 8-cfg localized current-current correlators 在 2026-08-22 spin-dagger/high-mode fix 前生成；最终 verification 要求重算，旧 arrays 已被 stale cleanup 移除。历史 job/result JSON 仅保留 provenance。
- 当前没有发现可验证的 genuine H–J–H C3 family；bounded search 不排除项目外部/private 数据存在。

## Measurement readiness

- canonical readiness 是 v4，不是 v3；它记录 `files_verified=true`、`ready=false`。
- charge/H–J–H blockers 是 approval authority/document、projectors、flavor weights、C2/C3/ratio formulas、time/contact/plateau policy 和 approved C2/C3 products。
- Current×Current 两点与 H–J–H charge normalization 是不同 observable 工作线；前者的 formula-neutral artifact/kernel 工作不自动解决后者的物理合同。

## 永久化边界

- 工作区/staging/remote 的当前状态只记录于 `TASKBOARD.md` 和 handover manifest，不进入稳定事实台账。
- Kunshan 大型数据不进入普通 Git；Git 保存小型 manifest、paths、hashes、axes 和 evidence summary。
- 当前 retention hard-gate 的动态状态见 `docs/data-retention-decision.json` 和 `TASKBOARD.md`；稳定规则是：只有 `approved` 或经用户签署的 `not-required` 才能通过 release-ready。
- 永久化与上线门禁见 `docs/permanence-and-deployment.md`；动态状态见 `TASKBOARD.md`。
