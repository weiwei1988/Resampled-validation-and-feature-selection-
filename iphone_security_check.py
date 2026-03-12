#!/usr/bin/env python3
"""
iPhone Security Diagnostic Tool
USB-C経由でMacとiPhoneを接続してセキュリティ診断を行うツール

使い方:
    1. iPhoneをUSB-CケーブルでMacに接続
    2. iPhoneの画面で「このコンピュータを信頼」をタップ
    3. python iphone_security_check.py を実行

必要なライブラリ:
    pip install pymobiledevice3
"""

import json
import re
import sys
import time
from datetime import datetime
from threading import Event, Thread

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

# 既知の不審/追跡ドメイン（一部）
SUSPICIOUS_DOMAINS = [
    "spy-phone",
    "flexispy",
    "mspy",
    "spyera",
    "highstermobile",
    "ikeymonitor",
    "phonetracker",
    "phonesheriff",
    "mobiletrackerapp",
]

# 診断ログ（グローバル）
syslog_entries = []
stop_syslog = Event()


def print_header():
    print()
    print("=" * 55)
    print("   iPhone Security Diagnostic Tool")
    print("   USB-C 接続によるiPhoneセキュリティ診断")
    print("=" * 55)
    print()


def check_dependencies():
    """必要なライブラリの確認"""
    try:
        import pymobiledevice3
        return True
    except ImportError:
        print("[エラー] pymobiledevice3 がインストールされていません。")
        print("  以下のコマンドでインストールしてください:")
        print("  pip install pymobiledevice3")
        return False


def connect_device():
    """iPhoneに接続してLockdownClientを返す"""
    from pymobiledevice3.lockdown import create_using_usbmux

    print("[接続中] iPhoneを検索しています...")
    try:
        lockdown = create_using_usbmux()
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
    info = {
        "device_name": lockdown.get_value("", "DeviceName"),
        "product_type": lockdown.get_value("", "ProductType"),
        "product_version": lockdown.get_value("", "ProductVersion"),
        "build_version": lockdown.get_value("", "BuildVersion"),
        "udid": lockdown.get_value("", "UniqueDeviceID"),
        "serial_number": lockdown.get_value("", "SerialNumber"),
    }
    return info


def check_installed_apps(lockdown):
    """インストール済みアプリの診断"""
    from pymobiledevice3.services.installation_proxy import InstallationProxyService

    print("[1/4] インストール済みアプリを診断中...")

    try:
        service = InstallationProxyService(lockdown=lockdown)
        apps = service.get_apps(app_types=["User", "System"])
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
                "description": KNOWN_SPYWARE[bundle_id]
            })
        if bundle_id in REMOTE_ACCESS_APPS:
            detected_remote.append({
                "bundle_id": bundle_id,
                "name": app_name,
                "description": REMOTE_ACCESS_APPS[bundle_id]
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

    return {
        "total": total_apps,
        "spyware": detected_spyware,
        "remote_access": detected_remote,
    }


def check_profiles(lockdown):
    """構成プロファイルの診断"""
    print("\n[2/4] 構成プロファイルを診断中...")

    try:
        from pymobiledevice3.services.mobile_config import MobileConfigService
        service = MobileConfigService(lockdown=lockdown)
        profiles = service.get_profile_list()
    except Exception as e:
        print(f"  [スキップ] プロファイル一覧の取得に失敗しました: {e}")
        return {"profiles": [], "suspicious": [], "error": str(e)}

    profile_list = profiles.get("ProfileMetadata", {})
    suspicious_profiles = []
    found_profiles = []

    for profile_id, profile_info in profile_list.items():
        name = profile_info.get("PayloadDisplayName", "不明")
        org = profile_info.get("PayloadOrganization", "不明")
        description = profile_info.get("PayloadDescription", "")

        found_profiles.append({
            "id": profile_id,
            "name": name,
            "organization": org,
            "description": description,
        })

        # MDMプロファイルや不審なプロファイルの検出
        is_suspicious = False
        reason = ""

        if "MDM" in name.upper() or "MDM" in org.upper():
            is_suspicious = True
            reason = "MDM（モバイルデバイス管理）プロファイル"
        elif org.lower() in ["unknown", "不明", ""]:
            is_suspicious = True
            reason = "発行元不明のプロファイル"

        if is_suspicious:
            suspicious_profiles.append({
                "name": name,
                "organization": org,
                "reason": reason,
            })

    print(f"  プロファイル数: {len(found_profiles)} 件")

    if suspicious_profiles:
        for p in suspicious_profiles:
            print(f"  [注意] '{p['name']}' (発行元: {p['organization']}) - {p['reason']}")
    else:
        print("  [OK] 不審な構成プロファイルは検出されませんでした")

    return {
        "profiles": found_profiles,
        "suspicious": suspicious_profiles,
    }


def collect_syslog_worker(lockdown, duration_sec):
    """バックグラウンドでsyslogを収集するワーカー"""
    global syslog_entries
    try:
        from pymobiledevice3.services.os_trace import OsTraceService
        service = OsTraceService(lockdown=lockdown)
        for entry in service.syslog():
            if stop_syslog.is_set():
                break
            syslog_entries.append(str(entry))
    except Exception:
        pass


def check_syslog(lockdown, duration_sec=20):
    """システムログの解析"""
    global syslog_entries
    syslog_entries = []
    stop_syslog.clear()

    print(f"\n[3/4] システムログを解析中（{duration_sec}秒間）...")
    print("  ※ログを収集中。しばらくお待ちください...")

    # バックグラウンドスレッドでsyslog収集
    worker = Thread(target=collect_syslog_worker, args=(lockdown, duration_sec), daemon=True)
    worker.start()

    # プログレス表示
    for i in range(duration_sec):
        time.sleep(1)
        remaining = duration_sec - i - 1
        print(f"  残り {remaining} 秒...", end="\r")

    stop_syslog.set()
    worker.join(timeout=3)
    print()

    # ログ解析
    ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    domain_pattern = re.compile(r'(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}')

    found_ips = set()
    suspicious_domains_found = []

    full_log = "\n".join(syslog_entries)

    # IPアドレス抽出（プライベートIPを除外）
    for ip in ip_pattern.findall(full_log):
        if not (ip.startswith("192.168.") or ip.startswith("10.") or
                ip.startswith("172.16.") or ip.startswith("127.") or
                ip == "0.0.0.0" or ip == "255.255.255.255"):
            found_ips.add(ip)

    # 不審ドメインの検出
    for domain in domain_pattern.findall(full_log):
        for sus_keyword in SUSPICIOUS_DOMAINS:
            if sus_keyword in domain.lower():
                suspicious_domains_found.append(domain)

    suspicious_domains_found = list(set(suspicious_domains_found))

    print(f"  収集したログエントリ: {len(syslog_entries)} 件")

    if found_ips:
        print(f"  外部通信先IP ({len(found_ips)} 件): {', '.join(list(found_ips)[:10])}")
    else:
        print("  外部通信先IP: 検出なし")

    if suspicious_domains_found:
        for domain in suspicious_domains_found:
            print(f"  [危険] 不審なドメインへの通信: {domain}")
    else:
        print("  [OK] 不審なドメインへの通信は検出されませんでした")

    return {
        "log_count": len(syslog_entries),
        "external_ips": list(found_ips),
        "suspicious_domains": suspicious_domains_found,
    }


def calculate_risk(app_result, profile_result, syslog_result):
    """危険度スコアを計算"""
    score = 0

    if "spyware" in app_result:
        score += len(app_result["spyware"]) * 30
    if "remote_access" in app_result:
        score += len(app_result["remote_access"]) * 10
    if "suspicious" in profile_result:
        score += len(profile_result["suspicious"]) * 20
    if "suspicious_domains" in syslog_result:
        score += len(syslog_result["suspicious_domains"]) * 25

    if score == 0:
        return "低", "特に問題は検出されませんでした"
    elif score < 30:
        return "中", "一部注意が必要な項目が見つかりました"
    else:
        return "高", "重大なリスクが検出されました。すぐに対処してください"


def save_report(device_info, app_result, profile_result, syslog_result, risk_level, risk_message):
    """診断レポートをJSONファイルに保存"""
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


def main():
    print_header()

    # 依存ライブラリ確認
    if not check_dependencies():
        sys.exit(1)

    # デバイス接続
    lockdown = connect_device()
    if lockdown is None:
        sys.exit(1)

    # デバイス情報取得
    device_info = get_device_info(lockdown)
    print(f"[接続成功] {device_info['device_name']} ({device_info['product_type']}, iOS {device_info['product_version']})")
    print(f"           UDID: {device_info['udid'][:8]}...{device_info['udid'][-4:]}")
    print()

    # 各診断の実行
    app_result = check_installed_apps(lockdown)
    profile_result = check_profiles(lockdown)
    syslog_result = check_syslog(lockdown, duration_sec=20)

    # 危険度算出
    risk_level, risk_message = calculate_risk(app_result, profile_result, syslog_result)

    # 結果サマリー
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

    # レポート保存
    report_file = save_report(device_info, app_result, profile_result, syslog_result, risk_level, risk_message)
    print(f"\n  詳細レポート保存先: {report_file}")
    print("=" * 55)
    print()
    print("[注意] このツールはジェイルブレイクなしで確認できる範囲のみ診断します。")
    print("       高度なスパイウェアは検出できない場合があります。")
    print()


if __name__ == "__main__":
    main()
