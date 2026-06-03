from __future__ import annotations

import re
from typing import List, Optional
from zoneinfo import available_timezones

from .grounding import has_technical_or_security_terms
from .models import ExtractionResult


EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
EMAIL_LIKE_RE = re.compile(r"[\w\.-]+@[\w\.-]+")
INDUSTRY_KEYWORDS = [
    "制造业",
    "互联网",
    "软件",
    "SaaS",
    "教育",
    "金融",
    "医疗",
    "医药",
    "零售",
    "消费品",
    "快消",
    "汽车",
    "地产",
    "物流",
    "出版",
]
PAIN_KEYWORDS = [
    ("漏跟进", "销售线索漏跟进"),
    ("漏客户", "销售线索漏跟进"),
    ("跟进不及时", "销售跟进不及时"),
    ("线索跟进慢", "销售线索跟进慢"),
    ("跟进慢", "销售线索跟进慢"),
    ("线索响应慢", "线索响应慢"),
    ("客户响应慢", "客户响应慢"),
    ("响应慢", "客户响应慢"),
    ("渠道效率低", "渠道效率低"),
    ("销售周期长", "销售周期长"),
    ("销售漏斗", "销售漏斗管理"),
    ("客户流失", "客户流失"),
    ("销售效率低", "销售效率低"),
    ("CRM 记录不完整", "CRM 记录不完整"),
    ("CRM记录不完整", "CRM 记录不完整"),
]
DEMO_KEYWORDS = ["demo", "Demo", "演示", "约个时间", "安排会议", "约下周", "约 Demo", "约Demo", "下周聊"]
DEMO_DECLINE_PATTERNS = [
    r"不\s*约\s*(?:demo|Demo|演示)?",
    r"先\s*不\s*约",
    r"暂时\s*不\s*约",
    r"不\s*安排\s*(?:demo|Demo|演示)?",
    r"先\s*不\s*安排",
    r"取消\s*(?:demo|Demo|演示)?",
    r"算了.*不\s*约",
    r"不用.*(?:demo|Demo|演示)",
    r"不需要.*(?:demo|Demo|演示)",
]
DEMO_CANCEL_PATTERNS = [
    r"(?:先|暂时|现在)?不\s*(?:约|安排|预约).{0,8}(?:demo|Demo|演示|会议)?",
    r"(?:demo|Demo|演示|会议).{0,8}(?:先|暂时|现在)?不\s*(?:约|安排|预约)",
    r"(?:取消|不用|不要).{0,8}(?:demo|Demo|演示|会议|预约)",
    r"算了.{0,8}不\s*(?:约|安排|预约)",
]
HIGH_RISK_KEYWORDS = {
    "合同": "CONTRACT_OR_PROCUREMENT",
    "法务": "LEGAL_OR_SECURITY",
    "安全审计": "LEGAL_OR_SECURITY",
    "安全问卷": "LEGAL_OR_SECURITY",
    "审计报告": "LEGAL_OR_SECURITY",
    "安全报告": "LEGAL_OR_SECURITY",
    "数据保护": "LEGAL_OR_SECURITY",
    "数据处理协议": "LEGAL_OR_SECURITY",
    "GDPR": "LEGAL_OR_SECURITY",
    "等保": "LEGAL_OR_SECURITY",
    "个保法": "LEGAL_OR_SECURITY",
    "个人信息保护法": "LEGAL_OR_SECURITY",
    "DPA": "LEGAL_OR_SECURITY",
    "SOC2": "LEGAL_OR_SECURITY",
    "ISO": "LEGAL_OR_SECURITY",
    "定制报价": "CUSTOM_QUOTE",
    "采购条款": "CONTRACT_OR_PROCUREMENT",
    "采购": "CONTRACT_OR_PROCUREMENT",
    "代采": "CONTRACT_OR_PROCUREMENT",
    "投诉": "COMPLAINT",
    "真人": "HUMAN_REQUEST",
    "人工": "HUMAN_REQUEST",
}
FEATURE_OBJECT_TERMS = [
    "产品",
    "系统",
    "工具",
    "平台",
    "AI",
    "CRM",
    "线索",
    "销售跟进",
    "自动分层",
    "提醒",
    "跟进建议",
    "集成",
    "对接",
]
FEATURE_INTENT_TERMS = [
    "支持",
    "功能",
    "能不能",
    "能否",
    "可以做",
    "能做",
    "能对接",
    "能集成",
    "怎么对接",
]
SCHEDULING_TERMS = ["下周", "明天", "后天", "上午", "下午", "几点", "约", "Demo", "会议", "聊"]
FEATURE_STRONG_OBJECT_TERMS = ["CRM", "系统", "产品", "功能", "集成", "对接", "线索", "销售跟进", "自动分层", "提醒"]


def extract_message(text: str) -> ExtractionResult:
    result = ExtractionResult()
    result.email = _extract_email(text)
    result.invalid_email = None if result.email else _extract_invalid_email(text)
    result.sales_team_size = _extract_sales_team_size(text)
    result.company_size = _extract_company_size(text, result.sales_team_size)
    result.industry = _extract_industry(text)
    result.pain_point = _extract_pain_point(text)
    result.budget = _extract_budget(text)
    result.budget_amount = _extract_budget_amount(text)
    result.budget_range = _extract_budget_range(text)
    if result.budget_amount and not result.budget:
        result.budget = result.budget_amount
    if result.budget_range and not result.budget:
        result.budget = result.budget_range
    result.timezone, result.ambiguous_timezone = _extract_timezone(text)
    result.demo_declined = _is_demo_declined(text)
    result.booking_cancelled = _is_booking_cancelled(text)
    result.demo_cancelled = result.demo_declined or result.booking_cancelled
    if result.demo_declined:
        result.demo_intent = False
        result.demo_purpose = None
    else:
        result.demo_intent = _has_any(text, DEMO_KEYWORDS)
        result.demo_purpose = _extract_demo_purpose(text, result.demo_intent)
    result.interest_area = _extract_interest_area(text)
    result.decision_maker = _extract_decision_maker(text)
    result.go_live_time = _extract_go_live_time(text)
    result.price_question = _is_price_question(text)
    result.metric_question = _is_metric_question(text)
    result.customer_case_question = _is_customer_case_question(text)
    result.technical_or_security_question = _is_technical_or_security_question(text)
    result.feature_question = _is_feature_question(text)
    result.prompt_injection = _is_prompt_injection(text)
    result.risk_flags = _extract_risk_flags(text)
    if result.ambiguous_timezone:
        result.risk_flags.append("AMBIGUOUS_TIMEZONE")
    if _has_timezone_offset_without_city(text):
        result.risk_flags.append("TIMEZONE_OFFSET_NEEDS_CITY")
    if result.invalid_email:
        result.risk_flags.append("INVALID_EMAIL_FORMAT")
    if result.demo_declined:
        result.risk_flags.append("BOOKING_DECLINED")
    if _has_industry_update(text, result.industry):
        result.risk_flags.append("INDUSTRY_UPDATED")
    return result


def extract_many(messages: List[str]) -> ExtractionResult:
    merged = ExtractionResult()
    for text in messages:
        item = extract_message(text)
        for field in [
            "company_size",
            "industry",
            "pain_point",
            "budget",
            "budget_amount",
            "budget_range",
            "decision_maker",
            "go_live_time",
            "email",
            "invalid_email",
            "timezone",
            "demo_purpose",
            "interest_area",
            "sales_team_size",
        ]:
            value = getattr(item, field)
            if value:
                setattr(merged, field, value)
        for flag in item.risk_flags:
            if flag not in merged.risk_flags:
                merged.risk_flags.append(flag)
        merged.demo_intent = merged.demo_intent or item.demo_intent
        merged.price_question = merged.price_question or item.price_question
        merged.metric_question = merged.metric_question or item.metric_question
        merged.customer_case_question = merged.customer_case_question or item.customer_case_question
        merged.technical_or_security_question = merged.technical_or_security_question or item.technical_or_security_question
        merged.feature_question = merged.feature_question or item.feature_question
        merged.prompt_injection = merged.prompt_injection or item.prompt_injection
        merged.ambiguous_timezone = merged.ambiguous_timezone or item.ambiguous_timezone
        merged.demo_cancelled = merged.demo_cancelled or item.demo_cancelled
        merged.demo_declined = merged.demo_declined or item.demo_declined
        merged.booking_cancelled = merged.booking_cancelled or item.booking_cancelled
    return merged


def _extract_email(text: str) -> Optional[str]:
    matches = list(EMAIL_RE.finditer(text))
    return matches[-1].group(0) if matches else None


def _extract_invalid_email(text: str) -> Optional[str]:
    for match in EMAIL_LIKE_RE.finditer(text):
        value = match.group(0).rstrip(".,，。;；")
        if "@" in value and not EMAIL_RE.fullmatch(value):
            return value
    return None


def _extract_sales_team_size(text: str) -> Optional[str]:
    patterns = [
        r"销售(?:团队|人员|同事)?\s*(?:有|只有|大概|大约|约|不到|将近|差不多)?\s*(\d+)\s*人",
        r"销售(?:团队|人员|同事)?\s*(?:规模|人数)?\s*(?:是|为|有)?\s*(\d+)\s*人",
        r"(\d+)\s*人\s*(?:的)?销售(?:团队|人员|同事)?",
        r"销售.{0,6}?(\d+)\s*人",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def _extract_company_size(text: str, sales_team_size: Optional[str]) -> Optional[str]:
    patterns = [
        r"我们(?:公司|企业|团队)?(?:是|有|大概|大约|约)?\s*(\d+)\s*人(?:公司|企业|团队|集团)?",
        r"(?:公司|企业|集团)(?:规模|人数)?(?:是|有|大概|大约|约)?\s*(\d+)\s*人",
        r"(\d+)\s*人(?:的)?(?:制造业|互联网|软件|教育|金融|医疗|零售)?(?:公司|企业|集团)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and match.group(1) != sales_team_size:
            return match.group(1)
    return None


def _extract_industry(text: str) -> Optional[str]:
    negated_spans = []
    for keyword in INDUSTRY_KEYWORDS:
        pattern = rf"(?:不是|不再是|已经不是|不做|不属于|不算)\s*{re.escape(keyword)}"
        negated_spans.extend(match.span() for match in re.finditer(pattern, text))

    mentions = []
    for keyword in INDUSTRY_KEYWORDS:
        for match in re.finditer(re.escape(keyword), text):
            if any(start <= match.start() < end for start, end in negated_spans):
                continue
            mentions.append((match.start(), keyword))
    if mentions:
        return max(mentions, key=lambda item: item[0])[1]
    return None


def _has_industry_update(text: str, extracted_industry: Optional[str]) -> bool:
    if not extracted_industry:
        return False
    has_negated_industry = any(
        re.search(rf"(?:不是|不再是|已经不是|不做|不属于|不算)\s*{re.escape(keyword)}", text)
        for keyword in INDUSTRY_KEYWORDS
    )
    return has_negated_industry


def _extract_pain_point(text: str) -> Optional[str]:
    pain_points: List[str] = []
    for keyword, normalized in PAIN_KEYWORDS:
        if keyword in text:
            if normalized not in pain_points:
                pain_points.append(normalized)
    return "；".join(pain_points) if pain_points else None


def _extract_budget(text: str) -> Optional[str]:
    if any(phrase in text for phrase in ["预算还没定", "预算未定", "还没定预算", "预算没有定", "预算没定", "还没定"]):
        return "not_set"
    budget_range = _extract_budget_range(text)
    if budget_range:
        return budget_range
    budget_amount = _extract_budget_amount(text)
    if budget_amount:
        return budget_amount
    if any(token in text for token in ["B轮", "B 轮", "A轮", "A 轮", "C轮", "C 轮"]) and any(token in text for token in ["采购", "预算", "评估"]):
        return "funding_stage_known"
    if "B-round" in text or "Series B" in text:
        return "funding_stage_known"
    if "有预算" in text or "预算明确" in text:
        return "known"
    return None


def _extract_budget_amount(text: str) -> Optional[str]:
    amount_match = re.search(r"(?:预算[^\d一二三四五六七八九十百千万两]{0,8})?(\d+(?:\.\d+)?)\s*万(?:元)?(?:.{0,12}预算)?", text)
    if amount_match and ("预算" in text or any(token in text for token in ["报价", "费用"])):
        return f"{amount_match.group(1)}万"
    return None


def _extract_budget_range(text: str) -> Optional[str]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|到|~|～)\s*(\d+(?:\.\d+)?)\s*万", text)
    if match and "预算" in text:
        return f"{match.group(1)}-{match.group(2)}万"
    return None


def _extract_timezone(text: str) -> tuple[Optional[str], bool]:
    if "CST" in text:
        return None, True
    for token in re.findall(r"\b[A-Za-z]+/[A-Za-z_]+(?:/[A-Za-z_]+)?\b", text):
        if token in available_timezones():
            return token, False
    zh_iana = re.search(r"(?:亚洲|亞州)/\s*(东京|東京|上海|新加坡)", text)
    if zh_iana:
        city = zh_iana.group(1)
        return {
            "东京": "Asia/Tokyo",
            "東京": "Asia/Tokyo",
            "上海": "Asia/Shanghai",
            "新加坡": "Asia/Singapore",
        }[city], False
    if "新加坡" in text:
        return "Asia/Singapore", False
    if any(keyword in text for keyword in ["上海", "北京", "中国时间", "北京时间"]):
        return "Asia/Shanghai", False
    timezone_aliases = {
        "东京": "Asia/Tokyo",
        "東京": "Asia/Tokyo",
        "日本": "Asia/Tokyo",
        "悉尼": "Australia/Sydney",
        "Sydney": "Australia/Sydney",
        "印度": "Asia/Kolkata",
        "美东": "America/New_York",
        "纽约": "America/New_York",
        "Eastern Time": "America/New_York",
        "美西": "America/Los_Angeles",
        "旧金山": "America/Los_Angeles",
        "Pacific Time": "America/Los_Angeles",
    }
    for keyword, timezone in timezone_aliases.items():
        if keyword in text:
            return timezone, False
    return None, False


def _extract_interest_area(text: str) -> Optional[str]:
    lowered = text.lower()
    if "CRM" in text or "crm" in lowered or "对接" in text or "集成" in text or "integration" in lowered:
        return "CRM 对接"
    if any(keyword in text for keyword in ["销售跟进", "线索跟进", "跟进线索", "漏跟进", "漏客户"]):
        return "销售线索跟进"
    return None


def _extract_demo_purpose(text: str, demo_intent: bool = True) -> Optional[str]:
    if not demo_intent:
        return None
    lowered = text.lower()
    if ("CRM" in text or "crm" in lowered) and (
        any(keyword in text for keyword in ["对接", "集成", "接入"]) or "integration" in lowered
    ):
        return "了解 CRM 对接"
    if any(keyword in text for keyword in ["线索跟进", "跟进线索", "漏跟进", "漏客户"]):
        return "了解销售线索跟进"
    purpose_match = re.search(r"(?:想看|了解|看看)\s*(?:一下|下)?\s*(?:你们|贵司|产品|系统|平台)?(?:的)?\s*([^，。；;！？?]{2,30})", text)
    if purpose_match:
        purpose = purpose_match.group(1).strip()
        purpose = re.sub(r"(?:能力|功能)?(?:怎么样|如何)$", "", purpose).strip()
        if purpose:
            return f"了解{purpose}"
    if any(keyword in text for keyword in ["想看", "了解", "看看"]):
        return "了解客户指定产品能力"
    return None


def _extract_decision_maker(text: str) -> Optional[str]:
    role_match = re.search(r"我是\s*(CEO|CFO|COO|CTO|CIO|CMO|VP|Owner|owner|负责人|销售负责人|采购负责人)", text, re.I)
    if role_match:
        return role_match.group(1).upper() if role_match.group(1).isalpha() else role_match.group(1)
    if any(keyword in text for keyword in ["我是负责人", "我负责", "决策人", "老板会看", "销售负责人", "由我决定", "我来决定", "我拍板"]):
        return "mentioned"
    return None


def _extract_go_live_time(text: str) -> Optional[str]:
    time_match = re.search(
        r"(Q[1-4]|FY\d{2,4}|下个月|本季度|这个季度|下季度|明年上半年|明年下半年|年底前|年内|圣诞节前|两周内|月底|本月|下周|尽快(?:上线|落地|试点)?)",
        text,
        re.I,
    )
    if time_match:
        return time_match.group(1)
    return None


def _has_any(text: str, keywords: List[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _has_demo_intent(text: str) -> bool:
    return _has_any(text, DEMO_KEYWORDS) and not _is_demo_declined(text)


def _is_demo_declined(text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in DEMO_DECLINE_PATTERNS)


def _is_booking_cancelled(text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in DEMO_CANCEL_PATTERNS)


def _is_demo_cancelled(text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in DEMO_CANCEL_PATTERNS)


def _has_timezone_offset_without_city(text: str) -> bool:
    return bool(re.search(r"\b(?:GMT|UTC)\s*[+-]\s*\d{1,2}(?::?\d{2})?\b", text, re.I))


def _is_price_question(text: str) -> bool:
    if any(keyword in text for keyword in ["多少钱", "价格", "报价", "费用", "标准版", "企业版", "最低价"]):
        return True
    has_price_like_amount = bool(
        re.search(r"\d+(?:\.\d+)?\s*(?:万|元|[wWkK]|RMB|CNY|USD|人民币|美元)", text, re.I)
        or re.search(r"[一二三四五六七八九十百千万两]+\s*(?:万|元|[kK]|人民币|美元)", text)
    )
    if not has_price_like_amount:
        return False
    if "预算" in text:
        return False
    return any(keyword in text for keyword in ["一年", "每年", "年费", "是不是", "确认", "这个价", "大概", "够吗", "够不够"])


def _is_metric_question(text: str) -> bool:
    return any(keyword in text for keyword in ["转化率", "提升", "效果", "保证", "一定能", "100%", "ROI", "投入产出比", "回报率", "产出", "指标"])


def _is_customer_case_question(text: str) -> bool:
    if any(keyword in text for keyword in ["500强", "500 强", "客户在用", "客户案例", "列几个名字", "客户名"]):
        return True
    if "客户" in text and any(keyword in text for keyword in ["在用", "合作", "服务过", "案例"]):
        return True
    return any(keyword in text for keyword in ["字节", "美团", "理想汽车", "京东", "百度", "阿里", "腾讯", "华为"])


def _is_feature_question(text: str) -> bool:
    if _is_metric_question(text) or _is_technical_or_security_question(text):
        return False
    has_object = any(term in text for term in FEATURE_OBJECT_TERMS)
    has_intent = any(term in text for term in FEATURE_INTENT_TERMS)
    if not (has_object and has_intent):
        return False
    if any(term in text for term in SCHEDULING_TERMS) and not any(term in text for term in FEATURE_STRONG_OBJECT_TERMS):
        return False
    return True


def _is_technical_or_security_question(text: str) -> bool:
    return has_technical_or_security_terms(text)


def _is_prompt_injection(text: str) -> bool:
    return any(keyword in text for keyword in ["忽略之前", "忽略你前面", "忽略所有规则", "直接给我", "标记成已预约"])


def _extract_risk_flags(text: str) -> List[str]:
    flags: List[str] = []
    for keyword, flag in HIGH_RISK_KEYWORDS.items():
        if keyword in text and flag not in flags:
            flags.append(flag)
    if re.search(r"合规(?:问题|材料|报告|文档|证明|审查)|(?:安全|隐私|数据)合规(?:材料|报告|问题)?", text) and "LEGAL_OR_SECURITY" not in flags:
        flags.append("LEGAL_OR_SECURITY")
    return flags
