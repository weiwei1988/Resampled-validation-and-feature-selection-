#!/usr/bin/env python3
"""
iPhone Security Diagnostic Tool (Enhanced)
USB-C経由でMacとiPhoneを接続してセキュリティ診断を行うツール

検出対象:
  - 既知ストーカーウェア / 商用スパイウェア (mSpy, FlexiSPY 等)
  - 高度なスパイウェア (Pegasus, Predator, Reign)
  - 不審な構成プロファイル / MDM
  - クラッシュログ内の不審プロセス痕跡
  - 不審な外部ドメインへの通信

使い方:
    1. iPhoneをUSB-CケーブルでMacに接続
    2. iPhoneの画面で「このコンピュータを信頼」をタップ
    3. venv/bin/python3 iphone_security_check.py

推奨: Amnesty International の MVT も併用してください
    pip install mvt
    mvt-ios check-backup --iocs ~/Downloads/iocs.stix2 <backup_path>
"""

import asyncio
import json
import re
import sys
from datetime import datetime

# ─────────────────────────────────────────────────────────
# 既知スパイウェア・ストーカーウェア バンドルIDリスト
# ─────────────────────────────────────────────────────────
KNOWN_SPYWARE = {
    "com.flexispy.flexispy":   "FlexiSPY（商用スパイウェア）",
    "com.mspy.mspy":           "mSpy（商用ストーカーウェア）",
    "com.highster.highster":   "Highster Mobile（商用スパイウェア）",
    "com.spyera.spyera":       "SPYERA（商用スパイウェア）",
    "com.phonesheriff.phonesheriff": "PhoneSheriff（監視ソフト）",
    "com.ikeymonitor.ikeymonitor":   "iKeyMonitor（キーロガー）",
    "com.familytime.familytime":     "FamilyTime（監視ソフト）",
    "com.spyzie.spyzie":       "Spyzie（商用スパイウェア）",
    "com.cocospy.cocospy":     "Cocospy（商用スパイウェア）",
    "com.minspy.minspy":       "Minspy（商用スパイウェア）",
}

REMOTE_ACCESS_APPS = {
    "com.anydesk.AnyDesk":               "AnyDesk（リモートデスクトップ）",
    "com.realvnc.viewer.ios":            "VNC Viewer（リモートデスクトップ）",
    "com.teamviewer.ios.TeamViewerHD":   "TeamViewer（リモートデスクトップ）",
    "com.splashtop.splashtop2":          "Splashtop（リモートデスクトップ）",
}

# ─────────────────────────────────────────────────────────
# 高度スパイウェア IOC（Indicators of Compromise）
# 出典: Amnesty International Security Lab / Citizen Lab 公開レポート
# ─────────────────────────────────────────────────────────

# Pegasus (NSO Group) — 公開済みドメインのサンプル
PEGASUS_DOMAINS = [
    "getpageone.com", "sirsticky.com", "cdn.getpageone.com",
    "tracfone-update.com", "fast-telemetry.com",
    "network-setup.com", "cdn.network-setup.com",
    "safe-connect.com", "appadvicecdn.com",
    "push-assets.com", "srvc-update.com",
    "pgssiphone.com", "pgss.info",
    "xvideos-static.com",           # Pegasus infrastructure (Citizen Lab 2021)
    "icloudprivaterelay.com",       # Pegasus masquerade domain
    "apple-icloud-relay.com",
    "googlr-services.com",          # Pegasus masquerade
    "applle-inc.com",
]

# Predator (Cytrox / Intellexa) — Citizen Lab 2022/2023 公開
PREDATOR_DOMAINS = [
    "analyticsplatform.co",
    "telecomsecurity.tech",
    "alwayschangingviews.com",
    "alwayschangingviews.net",
    "trackers-update.com",
    "cdn.trackers-update.com",
    "mobilesupportline.com",
    "phone-telemetry.com",
    "preloader.io",
    "intellexa-tech.com",
]

# Reign / QuaDream — Citizen Lab 2023 公開
REIGN_DOMAINS = [
    "cdn.cloudfront-assets.net",
    "cloudfront-assets.net",
    "imgloaderassets.com",
    "secure-cloudfront.net",
]

# 商用スパイウェア全般の既知C2ドメインパターン
COMMERCIAL_SPYWARE_DOMAINS = [
    "spy-phone", "flexispy", "mspy", "spyera",
    "highstermobile", "ikeymonitor", "phonetracker",
    "phonesheriff", "mobiletrackerapp", "spyzie",
    "cocospy", "minspy", "umobix", "eyezy",
    "spynger", "moniterro", "hoverwatch",
    "pctattoosoftware", "xnspy",
]

# 全不審ドメイン（syslog照合用）
ALL_SUSPICIOUS_DOMAINS = (
    PEGASUS_DOMAINS
    + PREDATOR_DOMAINS
    + REIGN_DOMAINS
    + COMMERCIAL_SPYWARE_DOMAINS
)

# ─────────────────────────────────────────────────────────
# 不審プロセス名（クラッシュログ照合用）
# ─────────────────────────────────────────────────────────
SUSPICIOUS_PROCESS_NAMES = [
    # Pegasus 関連プロセス（Amnesty MTD レポートより）
    "bh",       "fce",      "jbr",      "pce",
    "sbr",      "umh",      "launchrexd","wifid-helper",
    "msgacnt",  "ims",      "libtouchregd",
    "absinthe", "evasi0n",  "pangu",    "taig",   # ジェイルブレイクツール痕跡
    # 不審な擬似システムプロセス
    "locationd2", "cfprefssd2", "springboardd",
    "imagent2",   "wifid2",     "bluetoothd2",
]


def print_header():
    print()
    print("=" * 57)
    print("   iPhone Security Diagnostic Tool (Enhanced)")
    print("   USB-C 接続によるiPhoneセキュリティ診断")
    print("=" * 57)
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
    try:
        vals = lockdown.all_values
        return {
            "device_name":     vals.get("DeviceName", "Unknown"),
            "product_type":    vals.get("ProductType", "Unknown"),
            "product_version": vals.get("ProductVersion", "Unknown"),
            "build_version":   vals.get("BuildVersion", "Unknown"),
            "udid":            vals.get("UniqueDeviceID", getattr(lockdown, "udid", "Unknown")),
            "serial_number":   vals.get("SerialNumber", "Unknown"),
        }
    except Exception:
        return {
            "device_name":     getattr(lockdown, "name", "Unknown"),
            "product_type":    getattr(lockdown, "product_type", "Unknown"),
            "product_version": getattr(lockdown, "product_version", "Unknown"),
            "build_version":   getattr(lockdown, "build_version", "Unknown"),
            "udid":            getattr(lockdown, "udid", "Unknown"),
            "serial_number":   getattr(lockdown, "serial_number", "Unknown"),
        }


# ─────────────────────────────────────────────────────────
# [1/5] インストール済みアプリ診断
# ─────────────────────────────────────────────────────────
async def check_installed_apps(lockdown):
    from pymobiledevice3.services.installation_proxy import InstallationProxyService
    print("[1/5] インストール済みアプリを診断中...")

    try:
        async with InstallationProxyService(lockdown=lockdown) as service:
            try:
                apps = await service.get_apps(app_types=["User", "System"])
            except TypeError:
                apps = await service.get_apps()
    except Exception as e:
        print(f"  [スキップ] アプリ一覧の取得に失敗しました: {e}")
        return {"total": 0, "spyware": [], "remote_access": [], "error": str(e)}

    detected_spyware = []
    detected_remote = []

    for bundle_id, app_info in apps.items():
        name = app_info.get("CFBundleDisplayName", app_info.get("CFBundleName", bundle_id))
        if bundle_id in KNOWN_SPYWARE:
            detected_spyware.append({"bundle_id": bundle_id, "name": name,
                                     "description": KNOWN_SPYWARE[bundle_id]})
        if bundle_id in REMOTE_ACCESS_APPS:
            detected_remote.append({"bundle_id": bundle_id, "name": name,
                                    "description": REMOTE_ACCESS_APPS[bundle_id]})

    print(f"  確認済み: {len(apps)} アプリ")
    if detected_spyware:
        for a in detected_spyware:
            print(f"  [危険] {a['name']} ({a['bundle_id']}) - {a['description']}")
    else:
        print("  [OK] 既知スパイウェアは検出されませんでした")

    if detected_remote:
        for a in detected_remote:
            print(f"  [注意] {a['name']} ({a['bundle_id']}) - {a['description']}")
    else:
        print("  [OK] リモートアクセスアプリは検出されませんでした")

    return {"total": len(apps), "spyware": detected_spyware, "remote_access": detected_remote}


# ─────────────────────────────────────────────────────────
# [2/5] 構成プロファイル診断
# ─────────────────────────────────────────────────────────
async def check_profiles(lockdown):
    print("\n[2/5] 構成プロファイルを診断中...")
    try:
        from pymobiledevice3.services.mobile_config import MobileConfigService
        async with MobileConfigService(lockdown=lockdown) as service:
            profiles = await service.get_profile_list()
    except Exception as e:
        print(f"  [スキップ] プロファイル一覧の取得に失敗しました: {e}")
        return {"profiles": [], "suspicious": [], "error": str(e)}

    profile_list = profiles.get("ProfileMetadata", {})
    found = []
    suspicious = []

    for pid, pinfo in profile_list.items():
        name = pinfo.get("PayloadDisplayName", "不明")
        org  = pinfo.get("PayloadOrganization", "不明")
        found.append({"id": pid, "name": name, "organization": org})

        reason = ""
        if "MDM" in name.upper() or "MDM" in org.upper():
            reason = "MDM（モバイルデバイス管理）プロファイル"
        elif org.lower() in ["unknown", "不明", ""]:
            reason = "発行元不明のプロファイル"

        if reason:
            suspicious.append({"name": name, "organization": org, "reason": reason})

    print(f"  プロファイル数: {len(found)} 件")
    if suspicious:
        for p in suspicious:
            print(f"  [注意] '{p['name']}' (発行元: {p['organization']}) - {p['reason']}")
    else:
        print("  [OK] 不審な構成プロファイルは検出されませんでした")

    return {"profiles": found, "suspicious": suspicious}


# ─────────────────────────────────────────────────────────
# [3/5] クラッシュログ解析（高度スパイウェア検出の主要手法）
# ─────────────────────────────────────────────────────────
async def check_crash_logs(lockdown):
    print("\n[3/5] クラッシュログを解析中...")
    crash_log_text = ""

    try:
        from pymobiledevice3.services.crash_reports import CrashReportsManager
        async with CrashReportsManager(lockdown=lockdown) as manager:
            # クラッシュログ一覧を取得
            crash_entries = []
            async for item in manager.ls("/"):
                crash_entries.append(str(item))

            # 最新30件のログ内容を取得
            fetched = 0
            for entry in crash_entries[:50]:
                if fetched >= 30:
                    break
                try:
                    content = await manager.get(entry)
                    if isinstance(content, bytes):
                        content = content.decode("utf-8", errors="ignore")
                    crash_log_text += content + "\n"
                    fetched += 1
                except Exception:
                    continue

    except Exception as e:
        print(f"  [スキップ] クラッシュログの取得に失敗しました: {e}")
        return {"log_count": 0, "suspicious_processes": [], "jailbreak_traces": [], "error": str(e)}

    # 不審プロセス名の照合
    found_suspicious = []
    found_jailbreak = []
    jailbreak_keywords = ["jailbreak", "cydia", "substrate", "electra",
                          "unc0ver", "checkra1n", "evasi0n", "pangu", "taig"]

    for proc in SUSPICIOUS_PROCESS_NAMES:
        if re.search(r'\b' + re.escape(proc) + r'\b', crash_log_text, re.IGNORECASE):
            if proc in ["absinthe", "evasi0n", "pangu", "taig"]:
                found_jailbreak.append(proc)
            else:
                found_suspicious.append(proc)

    for kw in jailbreak_keywords:
        if kw in crash_log_text.lower() and kw not in found_jailbreak:
            found_jailbreak.append(kw)

    log_count = crash_log_text.count("Incident Identifier")
    print(f"  取得したクラッシュレポート: {log_count} 件")

    if found_suspicious:
        print(f"  [危険] 不審プロセスの痕跡: {', '.join(found_suspicious)}")
    else:
        print("  [OK] 不審プロセスの痕跡は検出されませんでした")

    if found_jailbreak:
        print(f"  [注意] ジェイルブレイク関連の痕跡: {', '.join(found_jailbreak)}")
        print("         ※ジェイルブレイク済み端末はスパイウェアに感染しやすい状態です")
    else:
        print("  [OK] ジェイルブレイクの痕跡は検出されませんでした")

    return {
        "log_count": log_count,
        "suspicious_processes": found_suspicious,
        "jailbreak_traces": found_jailbreak,
    }


# ─────────────────────────────────────────────────────────
# [4/5] システムログ解析（Pegasus/Predator IOCと照合）
# ─────────────────────────────────────────────────────────
async def check_syslog(lockdown, duration_sec=20):
    from pymobiledevice3.services.os_trace import OsTraceService
    print(f"\n[4/5] システムログを解析中（{duration_sec}秒間）...")
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

    ip_pat     = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    domain_pat = re.compile(r'(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}')

    found_ips       = set()
    sus_domains     = []
    pegasus_hits    = []
    predator_hits   = []
    reign_hits      = []
    full_log = "\n".join(entries)

    for ip in ip_pat.findall(full_log):
        if not ip.startswith(("192.168.", "10.", "172.16.", "127.")) \
                and ip not in ("0.0.0.0", "255.255.255.255"):
            found_ips.add(ip)

    for domain in domain_pat.findall(full_log):
        dl = domain.lower()
        for d in PEGASUS_DOMAINS:
            if d in dl:
                pegasus_hits.append(domain)
        for d in PREDATOR_DOMAINS:
            if d in dl:
                predator_hits.append(domain)
        for d in REIGN_DOMAINS:
            if d in dl:
                reign_hits.append(domain)
        for kw in COMMERCIAL_SPYWARE_DOMAINS:
            if kw in dl:
                sus_domains.append(domain)

    pegasus_hits  = list(set(pegasus_hits))
    predator_hits = list(set(predator_hits))
    reign_hits    = list(set(reign_hits))
    sus_domains   = list(set(sus_domains))

    print(f"  収集したログエントリ: {len(entries)} 件")
    if found_ips:
        print(f"  外部通信先IP ({len(found_ips)} 件): {', '.join(list(found_ips)[:10])}")

    if pegasus_hits:
        print(f"  [重大] Pegasus IOC 一致: {', '.join(pegasus_hits)}")
    if predator_hits:
        print(f"  [重大] Predator IOC 一致: {', '.join(predator_hits)}")
    if reign_hits:
        print(f"  [重大] Reign IOC 一致: {', '.join(reign_hits)}")
    if sus_domains:
        print(f"  [危険] 不審ドメインへの通信: {', '.join(sus_domains)}")
    if not (pegasus_hits or predator_hits or reign_hits or sus_domains):
        print("  [OK] 不審なドメインへの通信は検出されませんでした")

    return {
        "log_count": len(entries),
        "external_ips": list(found_ips),
        "pegasus_hits": pegasus_hits,
        "predator_hits": predator_hits,
        "reign_hits": reign_hits,
        "suspicious_domains": sus_domains,
    }


# ─────────────────────────────────────────────────────────
# [5/5] 診断情報（バッテリー・パフォーマンス異常）
# ─────────────────────────────────────────────────────────
async def check_diagnostics(lockdown):
    print("\n[5/5] 診断情報を確認中...")
    result = {}

    try:
        from pymobiledevice3.services.diagnostics import DiagnosticsService
        async with DiagnosticsService(lockdown=lockdown) as service:
            info = await service.info()
            battery = info.get("GasGauge", {})
            result["battery_level"]          = battery.get("BatteryCurrentCapacity", "N/A")
            result["battery_cycle_count"]     = battery.get("CycleCount", "N/A")
            result["battery_design_capacity"] = battery.get("DesignCapacity", "N/A")
            result["current_capacity"]        = battery.get("AbsoluteCapacity", "N/A")
    except Exception as e:
        result["error"] = str(e)

    if "error" not in result:
        print(f"  バッテリー残量: {result.get('battery_level', 'N/A')}%")
        print(f"  充放電サイクル数: {result.get('battery_cycle_count', 'N/A')}")
        print("  ※バッテリー消費が異常に速い場合、バックグラウンドで不審な処理が動いている可能性があります")
    else:
        print(f"  [スキップ] 診断情報の取得に失敗しました: {result['error']}")

    return result


# ─────────────────────────────────────────────────────────
# 危険度スコア算出
# ─────────────────────────────────────────────────────────
def calculate_risk(app_result, profile_result, crash_result, syslog_result):
    score = 0
    score += len(app_result.get("spyware", []))           * 30
    score += len(app_result.get("remote_access", []))     * 10
    score += len(profile_result.get("suspicious", []))    * 20
    score += len(crash_result.get("suspicious_processes", [])) * 25
    score += len(crash_result.get("jailbreak_traces", [])) * 15
    score += len(syslog_result.get("pegasus_hits", []))   * 50
    score += len(syslog_result.get("predator_hits", []))  * 50
    score += len(syslog_result.get("reign_hits", []))     * 50
    score += len(syslog_result.get("suspicious_domains", [])) * 20

    if score == 0:
        return "低", "特に問題は検出されませんでした"
    elif score < 30:
        return "中", "一部注意が必要な項目が見つかりました"
    else:
        return "高", "重大なリスクが検出されました。すぐに対処してください"


def save_report(device_info, app_result, profile_result,
                crash_result, syslog_result, diag_result,
                risk_level, risk_message):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"security_report_{timestamp}.json"
    report = {
        "generated_at": datetime.now().isoformat(),
        "device": device_info,
        "risk_level": risk_level,
        "risk_message": risk_message,
        "app_diagnosis": app_result,
        "profile_diagnosis": profile_result,
        "crash_log_diagnosis": crash_result,
        "syslog_diagnosis": syslog_result,
        "diagnostics": diag_result,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return filename


# ─────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────
async def main():
    print_header()

    if not check_dependencies():
        sys.exit(1)

    lockdown = await connect_device()
    if lockdown is None:
        sys.exit(1)

    device_info = get_device_info(lockdown)
    udid = device_info["udid"]
    print(f"[接続成功] {device_info['device_name']} "
          f"({device_info['product_type']}, iOS {device_info['product_version']})")
    print(f"           UDID: {udid[:8]}...{udid[-4:]}")
    print()

    app_result     = await check_installed_apps(lockdown)
    profile_result = await check_profiles(lockdown)
    crash_result   = await check_crash_logs(lockdown)
    syslog_result  = await check_syslog(lockdown, duration_sec=20)
    diag_result    = await check_diagnostics(lockdown)

    risk_level, risk_message = calculate_risk(
        app_result, profile_result, crash_result, syslog_result
    )

    print("\n" + "=" * 57)
    print("[完了] 診断結果サマリー")
    print(f"  危険度: {risk_level}")
    print(f"  {risk_message}")

    advanced_hits = (
        syslog_result.get("pegasus_hits", [])
        + syslog_result.get("predator_hits", [])
        + syslog_result.get("reign_hits", [])
    )
    if advanced_hits:
        print()
        print("  [重大警告] 高度スパイウェア（Pegasus/Predator/Reign）の痕跡が検出されました！")
        print("  以下の対応を強く推奨します:")
        print("  1. 端末を機内モードにして通信を遮断")
        print("  2. Amnesty International の MVT で詳細解析:")
        print("     pip install mvt")
        print("     mvt-ios check-backup <backup_path>")
        print("  3. Apple サポートまたはセキュリティ専門家に連絡")
        print("  4. 端末の完全初期化（工場出荷時リセット）")
    elif risk_level == "高":
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
        print("  2. MVT による詳細解析を推奨: pip install mvt")

    report_file = save_report(
        device_info, app_result, profile_result,
        crash_result, syslog_result, diag_result,
        risk_level, risk_message
    )
    print(f"\n  詳細レポート保存先: {report_file}")
    print("=" * 57)
    print()
    print("[注意] このツールはジェイルブレイクなしで確認できる範囲のみ診断します。")
    print("       完全な解析には Amnesty International の MVT を併用してください。")
    print("       https://github.com/mvt-project/mvt")
    print()


if __name__ == "__main__":
    asyncio.run(main())
