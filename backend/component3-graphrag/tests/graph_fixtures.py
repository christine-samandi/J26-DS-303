"""Act wording shared by the knowledge-graph and retrieval tests (abridged from the official
text of the 2017 Act and Amendments No. 10 of 2021, No. 45 of 2022 and No. 2 of 2025)."""

from entity_recognition.chunking import Chunk


def _c(cid, doc, text, section=None, schedule=None, paragraph=None, note=None):
    seg = (
        cid.replace("CHUNK-", "SEG-").rsplit("-", 1)[0]
        if section
        else cid.replace("CHUNK-", "SEG-")
    )
    return Chunk(cid, doc, seg, "section" if section else "schedule_paragraph", section, None,
                 schedule, paragraph, 1, None, note, 1, 1, text)  # fmt: skip


FIFTH = "Amendment of the Fifth Schedule to the principal enactment"
FIRST = "Amendment of the First Schedule to the principal enactment"

# Fifth Schedule para 2(a) in each Act (abridged from the official text)
RELIEF_CHUNKS = [
    _c("CHUNK-IRA-2017-SCH5-P002", "DOC-IRA-2017",
       "The reliefs referred to in section 52 shall be as follows: (a) Rs. 500,000 for each year "
       "of assessment, except that an individual who is a trustee ...; (b) in the case of an "
       "individual with income from employment, Rs. 700,000 for each year of assessment;",
       schedule="FIFTH SCHEDULE", paragraph="2"),
    _c("CHUNK-IRA-AMEND-10-2021-S055-2", "DOC-IRA-AMEND-10-2021",
       "(2) in paragraph 2 of that Schedule- (a) by the repeal of subparagraph (a) of that "
       "paragraph and the substitution therefor, of the following subparagraph: - “(a) (i) "
       "Rs. 500,000, for each year of assessment prior to January 1, 2020; and (ii) Rs. "
       "3,000,000, for each year of assessment commencing on or after January 1, 2020, except "
       "that an individual ...;”; (e) “(f) in the case of a resident individual, following "
       "expenditure up to a total sum of Rs. 1,200,000, incurred for a year of assessment”",
       section="55", note=FIFTH),
    _c("CHUNK-IRA-AMEND-45-2022-S036-2", "DOC-IRA-AMEND-45-2022",
       "(2) in paragraph 2 of that Schedule- (a) by the repeal of subparagraph (a) ...: - “(a) "
       "(i) Rs. 500,000, for each year of assessment prior to January 1, 2020; (ii) Rs. "
       "3,000,000, for each year of assessment commencing on or after January 1, 2020, but "
       "prior to April 1, 2022; (iii) Rs. 2,250,000, for first nine months and Rs. 300,000 for "
       "second three months of the year of assessment commencing on April 1, 2022; and (iv) "
       "Rs. 1,200,000, for each year of assessment commencing on or after April 1, 2023, "
       "except that ...;”", section="36", note=FIFTH),
    _c("CHUNK-IRA-AMEND-02-2025-S005-0", "DOC-IRA-AMEND-02-2025",
       "The Fifth Schedule to the principal enactment is hereby amended in subparagraph (a) of "
       "paragraph 2 as follows: -", section="5", note=FIFTH),
    _c("CHUNK-IRA-AMEND-02-2025-S005-3", "DOC-IRA-AMEND-02-2025",
       "(3) by the addition immediately after item (iv) of that subparagraph, of the following "
       "new item: - “(v) Rs. 1,800,000, for each year of assessment commencing on or after "
       "April 1, 2025,”.", section="5", note=FIFTH),
]  # fmt: skip

T17 = ("Not Exceeding Rs. 600,000 | 4% of the amount in excess of Rs. 0 Exceeding Rs. 600,000 "
       "but not exceeding Rs. 1,200,000 | Rs. 24,000 plus 8% of the amount in excess of "
       "Rs. 600,000 Exceeding Rs. 1,200,000 | Rs. 72,000 plus 12% of the amount")  # fmt: skip

SLAB_CHUNKS = [
    _c("CHUNK-IRA-2017-SCH1-P001-1", "DOC-IRA-2017",
       "(1) Subject to the provisions of subparagraph (2), the taxable income of a resident or "
       "non-resident individual for a year of assessment shall be taxed at the following "
       f"rates: Taxable Income | Tax Payable {T17}:", schedule="FIRST SCHEDULE", paragraph="1"),
    _c("CHUNK-IRA-AMEND-10-2021-S051-1", "DOC-IRA-AMEND-10-2021",
       "(1) in paragraph 1 of that Schedule- (b) ... “(1A) subject to the provisions of "
       "subparagraph (2), the taxable income of a resident or non-resident individual commencing "
       "from January 1, 2020 shall be taxed at the following rates: - Taxable income | Tax "
       "payable Not exceeding Rs. 3 million | 6% of the amount in excess of Rs.0 Exceeding "
       "Rs. 3 million | Rs. 180,000 plus 12% of the amount in excess of Rs. 3 million”; "
       "(iii) “(iii) on or after January 1, 2020: Total income from employment | Tax payable "
       "Not exceeding Rs. 10,000,000 | 0% of the amount in excess of Rs.0”",
       section="51", note=FIRST),
    _c("CHUNK-IRA-AMEND-45-2022-S033-1", "DOC-IRA-AMEND-45-2022",
       "“(1B) Subject to the provisions of subparagraph (2), the taxable income of a resident or "
       "nonresident individual for the year of assessment commencing from April 1, 2022 shall "
       "be taxed at the following rates: - (a) Taxable income for the first nine months period "
       "of the year of assessment commencing from April 1, 2022:- Taxable Income | Tax payable "
       "Not exceeding Rs. 2,250,000 | 6% of the amount in excess of Rs. 0 Exceeding Rs. "
       "2,250,000 | Rs. 135,000 plus 12% of the amount; (b) Taxable income for the second three "
       "months period of the year of assessment commencing from April 1, 2022:- Taxable Income "
       "| Tax payable Not exceeding Rs. 125,000 | 6% of the amount in excess of Rs.0 Exceeding "
       "Rs. 125,000 | Rs. 7,500 plus 12% of the amount”", section="33", note=FIRST),
]  # fmt: skip

OTHER_CHUNKS = [
    _c("CHUNK-IRA-2017-S052-2", "DOC-IRA-2017",
       "(2) In arriving at the taxable income of an individual who is resident in Sri Lanka for "
       "a year of assessment under section 3, the aggregate reliefs referred to in the Fifth "
       "Schedule to this Act shall be deducted.", section="52",
       note="Qualifying payments and reliefs."),
    _c("CHUNK-IRA-2017-S180-1", "DOC-IRA-2017",
       "(1) A person who fails to furnish a return of income shall be liable to a penalty equal "
       "to five per centum of the tax payable.", section="180", note="Penalty for late filing."),
]  # fmt: skip
