from fewrel.fewshot_re_kit.modular_encoder import (
    BERTBaseLayer,
    LukeBaseLayer,
    RobertaBaseLayer,
)

import pytest


@pytest.fixture
def example():
    raw_tokens = [
        "State",
        "Road",
        "66",
        "begins",
        "at",
        "the",
        "eastern",
        "end",
        "of",
        "a",
        "toll",
        "bridge",
        "over",
        "the",
        "Wabash",
        "River",
        "in",
        "New",
        "Harmony",
        "and",
        "ends",
        "at",
        "U.S.",
        "Route",
        "150",
        "east",
        "of",
        "Hardinsburg",
        ".",
    ]

    pos_head = [10, 11]
    pos_tail = [14, 15]

    return raw_tokens, pos_head, pos_tail


@pytest.mark.parametrize(
    "mask_entity, add_entity_token, ent1_tokens, ent2_tokens",
    [
        (
            True,
            True,
            ["[unused0]", "[unused4]", "[unused4]", "[unused2]"],
            ["[unused1]", "[unused4]", "[unused4]", "[unused3]"],
        ),
        (
            True,
            False,
            ['[unused4]', '[unused4]'],
            ['[unused4]', '[unused4]'],
        ),
        (
            False,
            True,
            ["[unused0]", "toll", "bridge", "[unused2]"],
            ["[unused1]", "wa", "##bas", "##h", "river", "[unused3]"],
        ),
        (
            False,
            False,
            ['toll', 'bridge'],
            ['wa', '##bas', '##h', 'river'],
        )
    ],
)
def test_bert_tokenizer(example, mask_entity, add_entity_token, ent1_tokens, ent2_tokens):
    raw_tokens, pos_head, pos_tail = example
    bert = BERTBaseLayer(
        pretrain_path="pretrain/bert-base-uncased",
        max_length=128,
        mask_entity=mask_entity,
        add_entity_token=add_entity_token,
    )
    indexed_tokens, ent1_index, ent2_index, _ = bert.tokenize(
        raw_tokens, pos_head, pos_tail
    )
    tokens = bert.tokenizer.convert_ids_to_tokens(indexed_tokens)
    assert tokens[ent1_index:ent1_index+ len(ent1_tokens)] == ent1_tokens
    assert tokens[ent2_index:ent2_index+ len(ent2_tokens)] == ent2_tokens

@pytest.mark.parametrize(
    "mask_entity, add_entity_token, ent1_tokens, ent2_tokens",
    [
        (
            True,
            True,
            ['[HEAD]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[HEAD]'],
            ['[TAIL]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[TAIL]']
        ),
        (
            True,
            False,
            ['[MASK]', '[MASK]', '[MASK]', '[MASK]'],
            ['[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]']
        ),
        (
            False,
            True,
            ['[HEAD]', 't', 'oll', 'Ġbridge', 'Ġ', '[HEAD]'],
            ['[TAIL]', 'W', 'ab', 'ash', 'ĠRiver', 'Ġ', "[TAIL]"]
        ),
        (
            False,
            False,
            ['t', 'oll', 'Ġbridge', 'Ġ'],
            ['W', 'ab', 'ash', 'ĠRiver', 'Ġ']
        )
    ],
)
def test_luke_tokenizer(example, mask_entity, add_entity_token, ent1_tokens, ent2_tokens):
    raw_tokens, pos_head, pos_tail = example
    luke = LukeBaseLayer(
        pretrain_path="pretrain/luke",
        max_length=128,
        mask_entity=mask_entity,
        add_entity_token=add_entity_token,
    )
    indexed_tokens, ent1_index, ent2_index, _ = luke.tokenize(
        raw_tokens, pos_head, pos_tail
    )
    tokens = luke.tokenizer.convert_ids_to_tokens(indexed_tokens)
    assert tokens[ent1_index:ent1_index+ len(ent1_tokens)] == ent1_tokens
    assert tokens[ent2_index:ent2_index+ len(ent2_tokens)] == ent2_tokens

@pytest.mark.parametrize(
    "mask_entity, add_entity_token, ent1_tokens, ent2_tokens",
    [
        (
            True,
            True,
            ['madeupword0000', '<mask>', '<mask>', 'madeupword0001'],
            # 'madeupword0003' is not in roberta's vocab and it's traslated to '<unk>'. 
            # But it shuld be ok since the trailing special token is not used in epresentation.
            ['madeupword0002', '<mask>', '<mask>', '<mask>', '<mask>', '<unk>']
        ),
        (
            True,
            False,
            ['<mask>', '<mask>'],
            ['<mask>', '<mask>', '<mask>', '<mask>']
        ),
        (
            False,
            True,
            ['madeupword0000', 'Ġtoll', 'Ġbridge', 'madeupword0001'],
            ['madeupword0002', 'ĠW', 'ab', 'ash', 'ĠRiver', '<unk>']
        ),
        (
            False,
            False,
            ['Ġtoll', 'Ġbridge'],
            ['ĠW', 'ab', 'ash', 'ĠRiver']
        )
    ],
)
def test_roberta_tokenizer(example, mask_entity, add_entity_token, ent1_tokens, ent2_tokens):
    raw_tokens, pos_head, pos_tail = example
    roberta = RobertaBaseLayer(
        pretrain_path="pretrain/luke",
        max_length=128,
        mask_entity=mask_entity,
        add_entity_token=add_entity_token,
    )
    indexed_tokens, ent1_index, ent2_index, _ = roberta.tokenize(
        raw_tokens, pos_head, pos_tail
    )
    tokens = roberta.tokenizer.convert_ids_to_tokens(indexed_tokens)
    assert tokens[ent1_index:ent1_index+ len(ent1_tokens)] == ent1_tokens
    assert tokens[ent2_index:ent2_index+ len(ent2_tokens)] == ent2_tokens
