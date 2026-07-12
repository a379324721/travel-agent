"""员工目录（mock）：模拟 SSO/HR 的身份查询，接口留好替换真实系统。"""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.travel.models import EmployeeGrade


@dataclass(frozen=True, slots=True)
class EmployeeProfile:
    employee_id: str
    name: str
    grade: EmployeeGrade
    department: str
    phone: str


_MOCK_EMPLOYEES: dict[str, EmployeeProfile] = {
    "E001": EmployeeProfile("E001", "王宁", EmployeeGrade.MANAGER, "技术部", "13800000001"),
    "E002": EmployeeProfile("E002", "李雷", EmployeeGrade.STAFF, "销售部", "13800000002"),
    "E003": EmployeeProfile("E003", "韩梅", EmployeeGrade.DIRECTOR, "产品部", "13800000003"),
}


class EmployeeDirectory:
    """按 user_id 返回员工档案；查不到返回 None（视为未登录/匿名）。"""

    def __init__(self, employees: dict[str, EmployeeProfile] | None = None) -> None:
        self._employees = employees or dict(_MOCK_EMPLOYEES)

    def get(self, user_id: str | None) -> EmployeeProfile | None:
        if not user_id:
            return None
        return self._employees.get(user_id)
