"""Small native GTK/WebKit window used by `kiln dashboard` on Linux."""

import sys
from urllib.parse import urlsplit

try:
    import gi

    gi.require_version("Gtk", "3.0")
    gi.require_version("WebKit2", "4.1")
    from gi.repository import Gtk, WebKit2
except (ImportError, ValueError) as error:
    print(
        "KILN's desktop window needs the GTK/WebKit runtime. On Kali/Debian run: "
        "sudo apt install python3-gi gir1.2-webkit2-4.1",
        file=sys.stderr,
    )
    print(f"Desktop runtime error: {error}", file=sys.stderr)
    sys.exit(2)


if len(sys.argv) != 2:
    print("Usage: kiln dashboard <local-url>", file=sys.stderr)
    sys.exit(2)

dashboard_url = sys.argv[1]
parsed = urlsplit(dashboard_url)
if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port:
    print("KILN desktop window only accepts its loopback dashboard URL.", file=sys.stderr)
    sys.exit(2)

origin = f"http://127.0.0.1:{parsed.port}"


def allow_local_navigation(_view, decision, decision_type):
    if decision_type != WebKit2.PolicyDecisionType.NAVIGATION_ACTION:
        return False
    request = decision.get_navigation_action().get_request()
    target = urlsplit(request.get_uri())
    target_origin = f"{target.scheme}://{target.netloc}"
    if target_origin == origin:
        decision.use()
    else:
        # Keep the application window scoped to KILN's local UI. External
        # destinations cannot replace the local app surface.
        decision.ignore()
    return True


def report_ui_loaded(view, event):
    if event == WebKit2.LoadEvent.FINISHED and view.get_title() == "KILN":
        print("KILN desktop UI loaded", flush=True)


Gtk.init(None)
window = Gtk.Window(title="KILN")
screen = window.get_screen()
monitor = screen.get_primary_monitor()
workarea = screen.get_monitor_workarea(monitor) if monitor >= 0 else None
screen_width = workarea.width if workarea else screen.get_width()
screen_height = workarea.height if workarea else screen.get_height()
window.set_default_size(min(1100, int(screen_width * 0.9)), min(740, int(screen_height * 0.9)))
window.set_size_request(min(800, screen_width), min(560, screen_height))
window.set_position(Gtk.WindowPosition.CENTER)
window.set_icon_name("kiln")
window.connect("destroy", Gtk.main_quit)

webview = WebKit2.WebView()
webview.connect("decide-policy", allow_local_navigation)
webview.connect("load-changed", report_ui_loaded)
webview.load_uri(dashboard_url)
window.add(webview)
window.show_all()
Gtk.main()
