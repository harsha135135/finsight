import pytest

EXAMPLE = {
    "filename": "ETR/2016/page_23.pdf",
    "id": "ETR/2016/page_23.pdf-2",
    "pre_text": ["entergy reported net revenue growth .", "the 2015 net revenue was $ 5.8 billion in total ."],
    "post_text": ["the increase was driven by retail pricing ."],
    "table_ori": [["", "Amount (In Millions)"], ["2014 net revenue", "$5,735"], ["2015 net revenue", "$5,829"]],
    "table": [["", "amount ( in millions )"], ["2014 net revenue", "$ 5735"], ["2015 net revenue", "$ 5829"]],
    "qa": {
        "question": "what is the growth rate in net revenue in 2015?",
        "program": "subtract(5829, 5735), divide(#0, 5735)",
        "exe_ans": 0.01639,
        "gold_inds": {"table_1": "...", "table_2": "..."},
    },
}


@pytest.fixture
def finqa_example():
    import copy
    return copy.deepcopy(EXAMPLE)
