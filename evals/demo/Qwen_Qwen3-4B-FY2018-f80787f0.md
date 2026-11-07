# 3M CO: FY2018

Source: 10-K accession 0001558370-19-000470, period ending 2018-12-31, filed 2019-02-07 ([original](https://www.sec.gov/Archives/edgar/data/66740/000155837019000470/mmm-20181231x10k.htm), sha256 `252d5b98ace7`, retrieved 2026-11-07T14:40:15.172098+00:00); amounts as reported in this filing

## Findings

| Metric | Period | Value | Status |
|---|---|---:|---|
| revenue | FY2018 | 32,765 million USD | accepted |
| operating\_margin | FY2018 | 22.0% | accepted |
| capex | FY2018 | -1,577 million USD | accepted |

## Calculations and sources

**revenue (FY2018)**
- revenue = $ 32,765 (millions, 2018) from passage `7019000470:item_8:t170r1`
- check: XBRL Revenues as reported in 0001558370-19-000470: 32765000000 \(agrees\)

**operating\_margin (FY2018)**
- operating\_income = 7,207 (millions, 2018) from passage `7019000470:item_8:t170r8`
- revenue = $ 32,765 (millions, 2018) from passage `7019000470:item_8:t170r1`
- divide(operating\_income, revenue) = 0.2199603235159468945521135358
- check: XBRL OperatingIncomeLoss / Revenues = 0.2200 \(agrees\)

**capex (FY2018)**
- capex = \(1,577\) (millions, 2018) from passage `7019000470:item_8:t175r18`
- check: XBRL PaymentsToAcquirePropertyPlantAndEquipment as reported in 0001558370-19-000470: 1577000000 \(agrees\)

## Unresolved and missing information

- none

## Run

- Outcome: completed
- Model: qwen Qwen/Qwen3-4B (contract finsight-report-v1)
- Generated: 2026-11-07T15:12:02.597868+00:00
- Usage: \{'attempts': 2, 'input\_tokens': '4379', 'output\_tokens': '839', 'cost\_usd': '0', 'unknown\_outcomes': 0\}
