from __future__ import annotations

import re
from typing import List, Optional

from .grounding import HIGH_RISK_CLAIM_TYPES, docs_from_tool_results, evidence_covers_facet
from .models import AgentResponse, AgentState, Claim, ToolCall, ToolResult, ValidationResult


PRICE_PATTERNS = [
    re.compile(r"\d+(?:\.\d+)?\s*万"),
    re.compile(r"\d+(?:\.\d+)?\s*元"),
    re.compile(r"¥\s*\d+"),
    re.compile(r"\$\s*\d+"),
    re.compile(r"[一二三四五六七八九十百千万两]+\s*(?:万|元|人民币|美元)"),
    re.compile(r"[一二三四五六七八九十百千万两]+\s*[kK]\b"),
    re.compile(r"\d+(?:\.\d+)?\s*[kK]\b"),
    re.compile(r"\d+(?:\.\d+)?\s*[wW]\b"),
    re.compile(r"\d+(?:\.\d+)?\s*(?:RMB|CNY|USD)\b", re.I),
    re.compile(r"\d+(?:\.\d+)?\s*(?:美元|人民币)"),
    re.compile(r"约\s*\d+(?:\.\d+)?\s*(?:万|元|[kKwW]|RMB|CNY|USD|美元|人民币)?", re.I),
]
METRIC_PATTERNS = [
    re.compile(r"提升\s*\d+%"),
    re.compile(r"转化率.{0,8}\d+%"),
    re.compile(r"保证"),
    re.compile(r"一定能"),
    re.compile(r"100%"),
]
FEATURE_PATTERNS = [
    "支持线索自动分层",
    "自动分层",
    "销售跟进提醒",
    "跟进提醒",
    "跟进建议",
    "生成跟进建议",
    "CRM 集成",
    "CRM集成",
    "与常见 CRM 集成",
    "对接 CRM",
    "产品支持",
]
FEATURE_PATTERN_FACETS = [
    ("支持线索自动分层", "lead_scoring"),
    ("自动分层", "lead_scoring"),
    ("销售跟进提醒", "follow_up_reminder"),
    ("跟进提醒", "follow_up_reminder"),
    ("跟进建议", "follow_up_suggestion"),
    ("生成跟进建议", "follow_up_suggestion"),
    ("CRM 集成", "crm_integration"),
    ("CRM集成", "crm_integration"),
    ("与常见 CRM 集成", "crm_integration"),
    ("对接 CRM", "crm_integration"),
]
BOOKED_PHRASES = ["已预约", "约好了", "已经帮你约", "Demo booked", "标成已预约"]
DISCOURAGING_PHRASES = ["先内部讨论", "之后再联系", "预算确定后再联系", "没有预算的话", "暂时不建议", "等你们预算明确"]
FAKE_CUSTOMER_PATTERNS = [
    "Alpha Corp",
    "Beta Group",
    "阿里",
    "腾讯",
    "华为",
    "字节",
    "美团",
    "理想汽车",
    "京东",
    "百度",
    "小米",
    "蔚来",
    "500 强客户",
    "多家 500",
]
CUSTOMER_CASE_CLAIM_PATTERNS = [
    re.compile(r"(?:客户|企业|公司).{0,12}(?:在用|合作|服务过)"),
    re.compile(r"(?:服务过|合作过).{0,12}(?:客户|企业|公司)"),
    re.compile(r"合作客户"),
    re.compile(r"500\s*强(?:客户|企业|公司)?"),
]
TECHNICAL_CLAIM_PATTERNS = [
    re.compile(r"\bRAG\b", re.I),
    re.compile(r"fine[-\s]?tuned|finetune", re.I),
    re.compile(r"微调|模型|大模型|核心架构|架构"),
    re.compile(r"p99|SLA|延迟|响应时间", re.I),
    re.compile(r"私有化|私有部署|本地部署"),
    re.compile(r"数据存(?:哪里|储|放)|训练数据|数据来源"),
    re.compile(r"DPA|SOC2|ISO|安全审计|审计报告", re.I),
]
ABSTAIN_CONTEXT_PATTERNS = [
    re.compile(r"没有(?:明确)?资料"),
    re.compile(r"不能.{0,8}确认"),
    re.compile(r"需要.{0,12}(?:同事|人工).{0,8}确认"),
    re.compile(r"需要.{0,12}(?:同事|人工).{0,8}提供"),
    re.compile(r"人工转接失败"),
    re.compile(r"转给.{0,12}(?:同事|人工)"),
]
TOOL_TO_ACTION = {
    "search_knowledge_base": "search_knowledge_base",
    "check_calendar": "check_calendar",
    "book_demo": "book_demo",
    "write_crm_note": "write_crm_note",
    "handoff_to_human": "handoff_to_human",
}


class ClaimValidator:
    @staticmethod
    def contains_price(text: str) -> bool:
        return any(pattern.search(text) for pattern in PRICE_PATTERNS)

    @staticmethod
    def contains_metric_guarantee(text: str) -> bool:
        return any(pattern.search(text) for pattern in METRIC_PATTERNS)

    @staticmethod
    def contains_booking_success_claim(text: str) -> bool:
        for phrase in BOOKED_PHRASES:
            for match in re.finditer(re.escape(phrase), text):
                context = text[max(0, match.start() - 10) : match.start()]
                if any(token in context for token in ["不会", "不能", "还不能", "未", "没有", "不要", "不把", "不能确认"]):
                    continue
                return True
        return False

    @staticmethod
    def contains_fake_customer_case(text: str) -> bool:
        return any(pattern in text for pattern in FAKE_CUSTOMER_PATTERNS)

    @staticmethod
    def contains_customer_case_claim(text: str) -> bool:
        return ClaimValidator.contains_fake_customer_case(text) or any(pattern.search(text) for pattern in CUSTOMER_CASE_CLAIM_PATTERNS)

    @staticmethod
    def contains_technical_claim(text: str) -> bool:
        if not any(pattern.search(text) for pattern in TECHNICAL_CLAIM_PATTERNS):
            return False
        return not any(pattern.search(text) for pattern in ABSTAIN_CONTEXT_PATTERNS)

    @staticmethod
    def contains_feature_claim(text: str) -> bool:
        return any(pattern in text for pattern in FEATURE_PATTERNS)

    @staticmethod
    def feature_facets(text: str) -> List[str]:
        facets: List[str] = []
        for pattern, facet in FEATURE_PATTERN_FACETS:
            if pattern in text and facet not in facets:
                facets.append(facet)
        return facets

    @staticmethod
    def contains_discouragement(text: str) -> bool:
        return any(phrase in text for phrase in DISCOURAGING_PHRASES)


class PreconditionValidator:
    @staticmethod
    def validate(tool_call: ToolCall, state: AgentState) -> ValidationResult:
        permission = PreconditionValidator.validate_allowed_action(tool_call, state)
        if not permission.ok:
            return permission
        if tool_call.tool_name == "book_demo":
            return PreconditionValidator.validate_book_demo(tool_call, state)
        if tool_call.tool_name == "check_calendar":
            return PreconditionValidator.validate_check_calendar(tool_call, state)
        if tool_call.tool_name == "write_crm_note":
            return PreconditionValidator.validate_write_crm_note(tool_call, state)
        return ValidationResult.pass_()

    @staticmethod
    def validate_allowed_action(tool_call: ToolCall, state: AgentState) -> ValidationResult:
        if tool_call.tool_name == "get_lead_context":
            return ValidationResult.pass_()
        expected_action = TOOL_TO_ACTION.get(tool_call.tool_name)
        if expected_action and expected_action not in state.allowed_actions:
            return ValidationResult.fail("TOOL_NOT_ALLOWED_IN_CURRENT_STATE")
        return ValidationResult.pass_()

    @staticmethod
    def validate_book_demo(tool_call: ToolCall, state: AgentState) -> ValidationResult:
        errors: List[str] = []
        if state.booking.book_demo_success:
            errors.append("DUPLICATE_BOOK_DEMO")
        if not state.collected_info.email:
            errors.append("MISSING_EMAIL")
        if not state.collected_info.timezone:
            errors.append("MISSING_TIMEZONE")
        if not state.collected_info.demo_purpose:
            errors.append("MISSING_DEMO_PURPOSE")
        if not state.booking.selected_slot_id:
            errors.append("MISSING_SELECTED_SLOT")
        if not state.booking.selected_slot_confirmed:
            errors.append("MISSING_EXPLICIT_SLOT_CONFIRMATION")
        offered_ids = [slot["slot_id"] for slot in state.booking.offered_slots]
        slot_id = tool_call.arguments.get("slot_id")
        if slot_id not in offered_ids:
            errors.append("UNKNOWN_SLOT_ID")
        elif slot_id != state.booking.selected_slot_id:
            errors.append("SELECTED_SLOT_MISMATCH")
        return ValidationResult.fail(*errors) if errors else ValidationResult.pass_()

    @staticmethod
    def validate_check_calendar(tool_call: ToolCall, state: AgentState) -> ValidationResult:
        timezone = tool_call.arguments.get("timezone")
        errors: List[str] = []
        if timezone is None:
            errors.append("MISSING_TIMEZONE")
        if timezone == "CST" or "AMBIGUOUS_TIMEZONE" in state.risk_flags:
            errors.append("AMBIGUOUS_TIMEZONE")
        if state.collected_info.timezone and timezone != state.collected_info.timezone:
            errors.append("TIMEZONE_MISMATCH")
        return ValidationResult.fail(*errors) if errors else ValidationResult.pass_()

    @staticmethod
    def validate_write_crm_note(tool_call: ToolCall, state: AgentState) -> ValidationResult:
        args = tool_call.arguments
        errors: List[str] = []
        summary = args.get("summary", "")
        next_action = args.get("next_action", "")
        if not summary:
            errors.append("CRM_SUMMARY_EMPTY")
        if args.get("qualification_level") not in ["high", "medium", "low", "unknown"]:
            errors.append("INVALID_QUALIFICATION_LEVEL")
        if not next_action:
            errors.append("CRM_NEXT_ACTION_EMPTY")
        if next_action == "Demo booked" and not state.booking.book_demo_success:
            errors.append("CRM_DEMO_BOOKED_WITHOUT_BOOK_DEMO")
        if "已预约" in summary and not state.booking.book_demo_success:
            errors.append("CRM_FALSE_BOOKING_STATUS")
        return ValidationResult.fail(*errors) if errors else ValidationResult.pass_()


class PostconditionValidator:
    @staticmethod
    def validate(message: str, state: AgentState, tool_results: List[ToolResult]) -> ValidationResult:
        errors: List[str] = []
        grounded_claims = list(state.last_claims or state.grounded_claims)
        docs_by_id = _docs_by_evidence_id(tool_results)
        current_evidence_ids = set(docs_by_id)
        for claim in grounded_claims:
            if claim.claim_type not in HIGH_RISK_CLAIM_TYPES:
                continue
            if not claim.evidence_ids:
                errors.append(_unsupported_error_for_claim(claim))
                continue
            if not set(claim.evidence_ids).issubset(current_evidence_ids):
                errors.append("EVIDENCE_FACET_MISMATCH")
                continue
            if _claim_uses_restricted_evidence(claim, docs_by_id):
                errors.append("RESTRICTED_EVIDENCE_DISCLOSED")
                continue
            if not _claim_evidence_covers_facet(claim, docs_by_id):
                errors.append("EVIDENCE_FACET_MISMATCH")
                continue
            if not evidence_covers_facet(state, claim.facet, claim.evidence_ids):
                errors.append("EVIDENCE_FACET_MISMATCH")

        if ClaimValidator.contains_booking_success_claim(message) and not state.booking.book_demo_success:
            errors.append("CLAIMED_BOOKED_WITHOUT_BOOK_DEMO")
        if ClaimValidator.contains_price(message) and not _has_grounded_claim(grounded_claims, "price", state, docs_by_id=docs_by_id):
            errors.append("UNSUPPORTED_PRICE_CLAIM")
        if ClaimValidator.contains_metric_guarantee(message) and not _has_grounded_claim(grounded_claims, "metric", state, docs_by_id=docs_by_id):
            errors.append("UNSUPPORTED_METRIC_CLAIM")
        if ClaimValidator.contains_customer_case_claim(message) and not _has_grounded_claim(grounded_claims, "customer_case", state, docs_by_id=docs_by_id):
            errors.append("UNSUPPORTED_CUSTOMER_CASE")
        feature_facets = ClaimValidator.feature_facets(message)
        if feature_facets:
            for facet in feature_facets:
                if not _has_grounded_claim(grounded_claims, "feature", state, facet=facet, docs_by_id=docs_by_id):
                    errors.append("UNSUPPORTED_FEATURE_CLAIM")
        elif ClaimValidator.contains_feature_claim(message) and not _has_grounded_claim(grounded_claims, "feature", state, docs_by_id=docs_by_id):
            errors.append("UNSUPPORTED_FEATURE_CLAIM")
        if ClaimValidator.contains_technical_claim(message) and not any(
            _has_grounded_claim(grounded_claims, claim_type, state, docs_by_id=docs_by_id)
            for claim_type in ["technical_architecture", "deployment", "security", "data_compliance"]
        ):
            errors.append("UNSUPPORTED_TECHNICAL_CLAIM")
        for result in tool_results:
            if result.tool_name == "write_crm_note":
                if result.arguments.get("next_action") == "Demo booked" and not state.booking.book_demo_success:
                    errors.append("CRM_DEMO_BOOKED_WITHOUT_BOOK_DEMO")
        return ValidationResult.fail(*errors) if errors else ValidationResult.pass_()


def _has_grounded_claim(
    claims: List[object],
    claim_type: str,
    state: AgentState,
    facet: Optional[str] = None,
    docs_by_id: Optional[dict] = None,
) -> bool:
    docs = docs_by_id or {}
    for claim in claims:
        if getattr(claim, "claim_type", None) != claim_type:
            continue
        if facet and getattr(claim, "facet", None) != facet:
            continue
        if not getattr(claim, "evidence_ids", None):
            continue
        if docs and (
            not set(getattr(claim, "evidence_ids", [])).issubset(set(docs))
            or _claim_uses_restricted_evidence(claim, docs)
            or not _claim_evidence_covers_facet(claim, docs)
        ):
            continue
        if evidence_covers_facet(state, getattr(claim, "facet", None), getattr(claim, "evidence_ids", [])):
            return True
    return False


def _docs_by_evidence_id(tool_results: List[ToolResult]) -> dict:
    docs = {}
    for doc in docs_from_tool_results(tool_results):
        evidence_id = doc.get("id")
        if evidence_id:
            docs[evidence_id] = doc
    return docs


def _claim_evidence_covers_facet(claim: object, docs_by_id: dict) -> bool:
    facet = getattr(claim, "facet", None)
    if not facet:
        return False
    for evidence_id in getattr(claim, "evidence_ids", []):
        doc = docs_by_id.get(evidence_id)
        if not doc or facet not in (doc.get("supported_facets") or []):
            return False
    return True


def _claim_uses_restricted_evidence(claim: object, docs_by_id: dict) -> bool:
    for evidence_id in getattr(claim, "evidence_ids", []):
        doc = docs_by_id.get(evidence_id)
        if doc and (doc.get("visibility") == "restricted" or doc.get("redacted")):
            return True
    return False


def _unsupported_error_for_claim(claim: Claim) -> str:
    if claim.claim_type == "price":
        return "UNSUPPORTED_PRICE_CLAIM"
    if claim.claim_type == "customer_case":
        return "UNSUPPORTED_CUSTOMER_CASE"
    if claim.claim_type == "metric":
        return "UNSUPPORTED_METRIC_CLAIM"
    if claim.claim_type == "feature":
        return "UNSUPPORTED_FEATURE_CLAIM"
    return "UNSUPPORTED_TECHNICAL_CLAIM"


class JsonSchemaValidator:
    @staticmethod
    def validate_response(response: AgentResponse) -> ValidationResult:
        try:
            response.model_dump(mode="json")
            return ValidationResult.pass_()
        except Exception as exc:  # pragma: no cover
            return ValidationResult.fail(str(exc))


class FallbackPolicy:
    @staticmethod
    def safe_message(errors: List[str], state: AgentState) -> str:
        if "HANDOFF_FAILED" in state.risk_flags:
            return "这类请求需要销售或安全同事接手。当前人工转接失败，我已经保留上下文，需要立即重试或通知销售负责人。"
        if "UNSUPPORTED_PRICE_CLAIM" in errors:
            return "具体价格需要销售同事根据你们的规模、使用场景和部署方式确认。我可以先记录需求，并请同事给你准确报价。"
        if "UNSUPPORTED_METRIC_CLAIM" in errors:
            return "效果需要结合你们的线索质量、销售流程和系统接入情况评估。我可以先记录场景，再请同事给出更准确的判断。"
        if "UNSUPPORTED_CUSTOMER_CASE" in errors:
            return "公开知识库里没有可公开的客户名称。我可以把你的案例需求记录下来，请销售同事提供可分享材料。"
        if "UNSUPPORTED_FEATURE_CLAIM" in errors:
            return "我现在没有查到足够可靠的产品资料来确认这个功能细节。为避免给你不准确的信息，我可以先记录你的需求，并请销售或产品同事确认后回复。"
        if any(error in ["UNSUPPORTED_TECHNICAL_CLAIM", "UNSUPPORTED_DEPLOYMENT_CLAIM", "UNSUPPORTED_SECURITY_CLAIM", "UNSUPPORTED_DATA_COMPLIANCE_CLAIM", "EVIDENCE_FACET_MISMATCH", "RESTRICTED_EVIDENCE_DISCLOSED"] for error in errors):
            return "这些问题涉及技术架构、性能指标、部署方式或数据合规，我不能在没有资料的情况下确认。我可以先记录下来，并请销售或技术同事确认。"
        if "CLAIMED_BOOKED_WITHOUT_BOOK_DEMO" in errors:
            if state.booking.status == "BOOKING_FAILED":
                return "刚才选择的时间不可用了，我不会确认预约成功。你可以换一个可选时间，或告诉我新的时间范围，我再重新查询。"
            return "我还不能确认 Demo 已经预约成功。请先补齐邮箱、时区、参会目的，并选择一个可用时间。"
        if "MISSING_EXPLICIT_SLOT_CONFIRMATION" in errors:
            return "我还不能预约 Demo。请明确确认一个可用时间后，我再继续预约。"
        if "AMBIGUOUS_TIMEZONE" in errors:
            return "CST 可能指多个时区。请确认你希望按哪个城市或 IANA 时区来安排，例如 Asia/Shanghai 或 America/Chicago。"
        if "TOOL_NOT_ALLOWED_IN_CURRENT_STATE" in errors:
            return "当前信息还不足以执行这个动作。我会先补齐必要信息或转入安全处理流程。"
        return "我需要再确认一下关键信息，才能安全地继续处理。"
