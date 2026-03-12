#!/usr/bin/env python3
"""
iPhone Security Diagnostic Tool
USB-C経由でMacとiPhoneを接続してセキュリティ診断を行うツール

使い方:
    1. iPhoneをUSB-CケーブルでMacに接続
    2. iPhoneの画面で「このコンピュータを信頼」をタップ
    3. venv/bin/python3 iphone_security_check.py を実行

必要なライブラリ:
    pip install pymobiledevice3
"""

import asyncio
import json
import re
import sys
from datetime import datetime

# 既知スパイウェア・ストーカーウェアのバンドルIDリスト
KNOWN_SPYWARE = {
    "com.flexispy.flexispy": "FlexiSPY（商用スパイウェア）",
    "com.mspy.mspy": "mSpy（商用ストーカーウェア）",
    "com.highster.highster": "Highster Mobile（商用スパイウェア）",
    "com.spyera.spyera": "SPYERA（商用スパイウェア）",
    "com.phonesheriff.phonesheriff": "PhoneSheriff（監視ソフト）",
    "com.ikeymonitor.ikeymonitor": "iKeyMonitor（キーロガー）",
    "com.familytime.familytime": "FamilyTime（監視ソフト）",
    "com.spyzie.spyzie": "Spyzie（商用スパイウェア）",
    "com.cocospy.cocospy": "Cocospy（商用スパイウェア）",
    "com.minspy.minspy": "Minspy（商用スパイウェア）",
}

# リモートアクセスアプリ（正規だが悪用される可能性あり）
REMOTE_ACCESS_APPS = {
    "com.anydesk.AnyDesk": "AnyDesk（リモートデスクトップ）",
    "com.realvnc.viewer.ios": "VNC Viewer（リモートデスクトップ）",
    "com.teamviewer.ios.TeamViewerHD": "TeamViewer（リモートデスクトップ）",
    "com.splashtop.splashtop2": "Splashtop（リモートデスクトップ）",
}

# 既知の不審/追跡ドメイン
SUSPICIOUS_DOMAINS = [
    "spy-phone", "flexispy", "mspy", "spyera",
    "highstermobile", "ikeymonitor", "phonetracker",
    "phonesheriff", "mobiletrackerapp",
]


def print_header():
    print()
    print("=" * 55)
    print("   iPhone Security Diagnostic Tool")
    print("   USB-C 接続によるiPhoneセキュリティ診断")
    print("=" * 55)
    print()


def check_dependencies():
    try:
        import pymobiledevice3  # noqa: F401
        return True
    except ImportError:
        print("[エラー] pymobiledevice3 がインストールされていません。")
        print("  venv/bin/pip install pymobiledevice3")
        return False


async def connect_device():
    """iPhoneに接続してLockdownClientを返す"""
    from pymobiledevice3.lockdown import create_using_usbmux

    print("[接続中] iPhoneを検索しています...")
    try:
        lockdown = await create_using_usbmux()
        return lockdown
    except Exception as e:
        print(f"[エラー] デバイスに接続できません: {e}")
        print()
        print("確認事項:")
        print("  1. iPhoneがUSB-Cで接続されているか")
        print("  2. iPhoneの画面で「このコンピュータを信頼」をタップしたか")
        print("  3. iPhoneのロックが解除されているか")
        return None


def get_device_info(lockdown):
    """デバイス基本情報の取得"""
    # pymobiledevice3 v4+ はプロパティ経由でアクセス
    try:
        vals = lockdown.all_values
        return {
            "device_name":     vals.get("DeviceName", "Unknown"),
            "product_type":    vals.get("ProductType", "Unknown"),
            "product_version": vals.get("ProductVersion", "Unknown"),
            "build_version":   vals.get("BuildVersion", "Unknown"),
            "udid":            vals.get("UniqueDeviceID", lockdown.udid),
            "serial_number":   vals.get("SerialNumber", "Unknown"),
        }
    except Exception:
        # フォールバック: 属性で直接アクセス
        return {
            "device_name":     getattr(lockdown, "name", "Unknown"),
            "product_type":    getattr(lockdown, "product_type", "Unknown"),
            "product_version": getattr(lockdown, "product_version", "Unknown"),
            "build_version":   getattr(lockdown, "build_version", "Unknown"),
            "udid":            getattr(lockdown, "udid", "Unknown"),
            "serial_number":   getattr(lockdown, "serial_number", "Unknown"),
        }


async def check_installed_apps(lockdown):
    """インストール済みアプリの診断"""
    from pymobiledevice3.services.installation_proxy import InstallationProxyService

    print("[1/4] インストール済みアプリを診断中...")

    try:
        async with InstallationProxyService(lockdown=lockdown) as service:
            apps = await service.get_apps(app_types=["User", "System"])
    except TypeError:
        # 古いAPIシグネチャのフォールバック
        try:
            async with InstallationProxyService(lockdown=lockdown) as service:
                apps = await service.get_apps()
        except Exception as e:
            print(f"  [スキップ] アプリ一覧の取得に失敗しました: {e}")
            return {"total": 0, "spyware": [], "remote_access": [], "error": str(e)}
    except Exception as e:
        print(f"  [スキップ] アプリ一覧の取得に失敗しました: {e}")
        return {"total": 0, "spyware": [], "remote_access": [], "error": str(e)}

    total_apps = len(apps)
    detected_spyware = []
    detected_remote = []

    for bundle_id, app_info in apps.items():
        app_name = app_info.get("CFBundleDisplayName", app_info.get("CFBundleName", bundle_id))
        if bundle_id in KNOWN_SPYWARE:
            detected_spyware.append({
                "bundle_id": bundle_id,
                "name": app_name,
                "description": KNOWN_SPYWARE[bundle_id],
            })
        if bundle_id in REMOTE_ACCESS_APPS:
            detected_remote.append({
                "bundle_id": bundle_id,
                "name": app_name,
                "description": REMOTE_ACCESS_APPS[bundle_id],
            })

    print(f"  確認済み: {total_apps} アプリ")

    if detected_spyware:
        for app in detected_spyware:
            print(f"  [危険] {app['name']} ({app['bundle_id']}) - {app['description']}")
    else:
        print("  [OK] 既知スパイウェアは検出されませんでした")

    if detected_remote:
        for app in detected_remote:
            print(f"  [注意] {app['name']} ({app['bundle_id']}) - {app['description']}")
    else:
        print("  [OK] リモートアクセスアプリは検出されませんでした")

    return {"total": total_apps, "spyware": detected_spyware, "remote_access": detected_remote}


async def check_profiles(lockdown):
    """構成プロファイルの診断"""
    print("\n[2/4] 構成プロファイルを診断中...")

    try:
        from pymobiledevice3.services.mobile_config import MobileConfigService
        async with MobileConfigService(lockdown=lockdown) as service:
            profiles = await service.get_profile_list()
    except Exception as e:
        print(f"  [スキップ] プロファイル一覧の取得に失敗しました: {e}")
        return {"profiles": [], "suspicious": [], "error": str(e)}

    profile_list = profiles.get("ProfileMetadata", {})
    found_profiles = []
    suspicious_profiles = []

    for profile_id, profile_info in profile_list.items():
        name = profile_info.get("PayloadDisplayName", "不明")
        org  = profile_info.get("PayloadOrganization", "不明")
        found_profiles.append({"id": profile_id, "name": name, "organization": org})

        reason = ""
        if "MDM" in name.upper() or "MDM" in org.upper():
            reason = "MDM（モバイルデバイス管理）プロファイル"
        elif org.lower() in ["unknown", "不明", ""]:
            reason = "発行元不明のプロファイル"

        if reason:
            suspicious_profiles.append({"name": name, "organization": org, "reason": reason})

    print(f"  プロファイル数: {len(found_profiles)} 件")

    if suspicious_profiles:
        for p in suspicious_profiles:
            print(f"  [注意] '{p['name']}' (発行元: {p['organization']}) - {p['reason']}")
    else:
        print("  [OK] 不審な構成プロファイルは検出されませんでした")

    return {"profiles": found_profiles, "suspicious": suspicious_profiles}


async def check_syslog(lockdown, duration_sec=20):
    """システムログの解析（asyncio.wait_for でタイムアウト）"""
    from pymobiledevice3.services.os_trace import OsTraceService

    print(f"\n[3/4] システムログを解析中（{duration_sec}秒間）...")
    print("  ※ログを収集中。しばらくお待ちください...")

    entries = []

    async def collect():
        try:
            async with OsTraceService(lockdown=lockdown) as service:
                async for entry in service.syslog():
                    entries.append(str(entry))
        except Exception:
            pass

    try:
        await asyncio.wait_for(collect(), timeout=duration_sec)
    except asyncio.TimeoutError:
        pass

    # ログ解析
    ip_pattern     = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    domain_pattern = re.compile(r'(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}')

    found_ips = set()
    sus_domains = []
    full_log = "\n".join(entries)

    for ip in ip_pattern.findall(full_log):
        if not (ip.startswith(("192.168.", "10.", "172.16.", "127.")) or
                ip in ("0.0.0.0", "255.255.255.255")):
            found_ips.add(ip)

    for domain in domain_pattern.findall(full_log):
        for kw in SUSPICIOUS_DOMAINS:
            if kw in domain.lower():
                sus_domains.append(domain)

    sus_domains = list(set(sus_domains))

    print(f"  収集したログエントリ: {len(entries)} 件")
    if found_ips:
        print(f"  外部通信先IP ({len(found_ips)} 件): {', '.join(list(found_ips)[:10])}")
    else:
        print("  外部通信先IP: 検出なし")

    if sus_domains:
        for d in sus_domains:
            print(f"  [危険] 不審なドメインへの通信: {d}")
    else:
        print("  [OK] 不審なドメインへの通信は検出されませんでした")

    return {"log_count": len(entries), "external_ips": list(found_ips), "suspicious_domains": sus_domains}


def calculate_risk(app_result, profile_result, syslog_result):
    score = 0
    score += len(app_result.get("spyware", [])) * 30
    score += len(app_result.get("remote_access", [])) * 10
    score += len(profile_result.get("suspicious", [])) * 20
    score += len(syslog_result.get("suspicious_domains", [])) * 25

    if score == 0:
        return "低", "特に問題は検出されませんでした"
    elif score < 30:
        return "中", "一部注意が必要な項目が見つかりました"
    else:
        return "高", "重大なリスクが検出されました。すぐに対処してください"


def save_report(device_info, app_result, profile_result, syslog_result, risk_level, risk_message):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"security_report_{timestamp}.json"
    report = {
        "generated_at": datetime.now().isoformat(),
        "device": device_info,
        "risk_level": risk_level,
        "risk_message": risk_message,
        "app_diagnosis": app_result,
        "profile_diagnosis": profile_result,
        "syslog_diagnosis": {
            "log_count": syslog_result.get("log_count", 0),
            "external_ips": syslog_result.get("external_ips", []),
            "suspicious_domains": syslog_result.get("suspicious_domains", []),
        },
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return filename


async def main():
    print_header()

    if not check_dependencies():
        sys.exit(1)

    lockdown = await connect_device()
    if lockdown is None:
        sys.exit(1)

    device_info = get_device_info(lockdown)
    print(f"[接続成功] {device_info['device_name']} ({device_info['product_type']}, iOS {device_info['product_version']})")
    udid = device_info['udid']
    print(f"           UDID: {udid[:8]}...{udid[-4:]}")
    print()

    app_result     = await check_installed_apps(lockdown)
    profile_result = await check_profiles(lockdown)
    syslog_result  = await check_syslog(lockdown, duration_sec=20)

    risk_level, risk_message = calculate_risk(app_result, profile_result, syslog_result)

    print("\n" + "=" * 55)
    print("[4/4] 診断完了")
    print(f"  危険度: {risk_level}")
    print(f"  {risk_message}")

    if risk_level == "高":
        print()
        print("  推奨対応:")
        print("  1. 不審なアプリをすぐにアンインストール")
        print("  2. 不審なプロファイルを削除（設定 > 一般 > VPNとデバイス管理）")
        print("  3. Appleサポートまたはセキュリティ専門家に相談")
        print("  4. 最終手段: 端末の完全初期化（工場出荷時リセット）")
    elif risk_level == "中":
        print()
        print("  推奨対応:")
        print("  1. 不審なアプリ・プロファイルを確認・削除")
        print("  2. しばらく様子を見て再度診断を実施")

    report_file = save_report(device_info, app_result, profile_result, syslog_result, risk_level, risk_message)
    print(f"\n  詳細レポート保存先: {report_file}")
    print("=" * 55)
    print()
    print("[注意] このツールはジェイルブレイクなしで確認できる範囲のみ診断します。")
    print("       高度なスパイウェアは検出できない場合があります。")
    print()


if __name__ == "__main__":
    asyncio.run(main())
