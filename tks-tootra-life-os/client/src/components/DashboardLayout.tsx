import { useAuth } from "@/_core/hooks/useAuth";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import { getLoginUrl } from "@/const";
import { useIsMobile } from "@/hooks/useMobile";
import {
  Inbox,
  CalendarCheck,
  Zap,
  BarChart3,
  Grid3X3,
  Shield,
  ClipboardList,
  LogOut,
  PanelLeft,
  Triangle,
  Flame,
  HelpCircle,
} from "lucide-react";
import { CSSProperties, useEffect, useRef, useState } from "react";
import { useLocation } from "wouter";
import { DashboardLayoutSkeleton } from "./DashboardLayoutSkeleton";
import { Button } from "./ui/button";
import { trpc } from "@/lib/trpc";
import { resetOnboarding } from "@/components/OnboardingSlideshow";

const menuItems = [
  { icon: Inbox, label: "Inbox", path: "/" },
  { icon: CalendarCheck, label: "Tasks", path: "/tasks" },
  { icon: Zap, label: "Equations", path: "/equations" },
  { icon: Grid3X3, label: "Heatmap", path: "/heatmap" },
  { icon: Shield, label: "Power Gaps", path: "/power" },
  { icon: BarChart3, label: "Weekly Review", path: "/review" },
  { icon: ClipboardList, label: "D/W/P Tracker", path: "/dwp" },
];

const SIDEBAR_WIDTH_KEY = "sidebar-width";
const DEFAULT_WIDTH = 260;
const MIN_WIDTH = 200;
const MAX_WIDTH = 420;

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    const saved = localStorage.getItem(SIDEBAR_WIDTH_KEY);
    return saved ? parseInt(saved, 10) : DEFAULT_WIDTH;
  });
  const { loading, user } = useAuth();

  useEffect(() => {
    localStorage.setItem(SIDEBAR_WIDTH_KEY, sidebarWidth.toString());
  }, [sidebarWidth]);

  if (loading) return <DashboardLayoutSkeleton />;

  if (!user) {
    return (
      <div className="flex items-center justify-center min-h-screen tks-triangle-bg">
        <div className="flex flex-col items-center gap-8 p-10 max-w-md w-full">
          {/* Triangle Logo */}
          <div className="relative">
            <div className="w-20 h-20 flex items-center justify-center">
              <Triangle className="w-16 h-16 text-tks-gold fill-tks-gold/10" />
            </div>
          </div>
          <div className="flex flex-col items-center gap-3">
            <h1 className="text-3xl font-bold tracking-tight text-center text-foreground">
              TKS Tootra
            </h1>
            <p className="text-sm text-muted-foreground text-center italic">
              Est Ars Celare Artem
            </p>
            <p className="text-sm text-muted-foreground text-center max-w-sm mt-2">
              Your Life Execution System powered by the Total Knowledge System.
              Sign in to begin mapping your life with intentionality.
            </p>
          </div>
          <Button
            onClick={() => { window.location.href = getLoginUrl(); }}
            size="lg"
            className="w-full bg-tks-gold hover:bg-tks-gold-light text-background font-semibold shadow-lg hover:shadow-xl transition-all"
          >
            Sign in to Continue
          </Button>
        </div>
      </div>
    );
  }

  return (
    <SidebarProvider style={{ "--sidebar-width": `${sidebarWidth}px` } as CSSProperties}>
      <DashboardLayoutContent setSidebarWidth={setSidebarWidth}>
        {children}
      </DashboardLayoutContent>
    </SidebarProvider>
  );
}

function DashboardLayoutContent({
  children,
  setSidebarWidth,
}: {
  children: React.ReactNode;
  setSidebarWidth: (width: number) => void;
}) {
  const { user, logout } = useAuth();
  const [location, setLocation] = useLocation();
  const { state, toggleSidebar } = useSidebar();
  const isCollapsed = state === "collapsed";
  const [isResizing, setIsResizing] = useState(false);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const activeMenuItem = menuItems.find(item => item.path === location);
  const isMobile = useIsMobile();

  // Mode toggle
  const modeQuery = trpc.settings.getMode.useQuery();
  const modeMutation = trpc.settings.setMode.useMutation({
    onSuccess: () => modeQuery.refetch(),
  });
  const currentMode = modeQuery.data?.mode || "life_os";
  const isWorkSprint = currentMode === "work_sprint";

  // Filter menu items based on mode
  const visibleItems = isWorkSprint
    ? menuItems.filter(i => ["/", "/tasks"].includes(i.path))
    : menuItems;

  useEffect(() => {
    if (isCollapsed) setIsResizing(false);
  }, [isCollapsed]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;
      const sidebarLeft = sidebarRef.current?.getBoundingClientRect().left ?? 0;
      const newWidth = e.clientX - sidebarLeft;
      if (newWidth >= MIN_WIDTH && newWidth <= MAX_WIDTH) setSidebarWidth(newWidth);
    };
    const handleMouseUp = () => setIsResizing(false);
    if (isResizing) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    }
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [isResizing, setSidebarWidth]);

  return (
    <>
      <div className="relative" ref={sidebarRef}>
        <Sidebar collapsible="icon" className="border-r-0" disableTransition={isResizing}>
          <SidebarHeader className="h-16 justify-center">
            <div className="flex items-center gap-3 px-2 transition-all w-full">
              <button
                onClick={toggleSidebar}
                className="h-8 w-8 flex items-center justify-center hover:bg-accent rounded-lg transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring shrink-0"
                aria-label="Toggle navigation"
              >
                <PanelLeft className="h-4 w-4 text-muted-foreground" />
              </button>
              {!isCollapsed && (
                <div className="flex items-center gap-2 min-w-0">
                  <Triangle className="h-5 w-5 text-tks-gold fill-tks-gold/20 shrink-0" />
                  <span className="font-bold tracking-tight truncate text-foreground">
                    TKS Tootra
                  </span>
                </div>
              )}
            </div>
          </SidebarHeader>

          <SidebarContent className="gap-0">
            <SidebarMenu className="px-2 py-1">
              {visibleItems.map(item => {
                const isActive = location === item.path;
                return (
                  <SidebarMenuItem key={item.path}>
                    <SidebarMenuButton
                      isActive={isActive}
                      onClick={() => setLocation(item.path)}
                      tooltip={item.label}
                      className="h-10 transition-all font-normal"
                    >
                      <item.icon className={`h-4 w-4 ${isActive ? "text-tks-gold" : ""}`} />
                      <span>{item.label}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>

            {/* How to Use */}
            <SidebarMenu className="px-2 py-1 mt-2 border-t border-border/30 pt-2">
              <SidebarMenuItem>
                <SidebarMenuButton
                  onClick={() => {
                    resetOnboarding();
                    window.dispatchEvent(new CustomEvent("tks-show-onboarding"));
                  }}
                  tooltip="How to Use"
                  className="h-10 transition-all font-normal text-muted-foreground hover:text-foreground"
                >
                  <HelpCircle className="h-4 w-4" />
                  <span>How to Use</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>

            {/* Mode Toggle */}
            {!isCollapsed && (
              <div className="px-4 mt-4">
                <button
                  onClick={() => modeMutation.mutate({ mode: isWorkSprint ? "life_os" : "work_sprint" })}
                  className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-all border border-border hover:border-tks-gold/30"
                >
                  {isWorkSprint ? (
                    <>
                      <Flame className="h-3.5 w-3.5 text-tks-flame" />
                      <span className="text-tks-flame">Work Sprint</span>
                      <span className="ml-auto text-muted-foreground">Switch to Life OS</span>
                    </>
                  ) : (
                    <>
                      <Triangle className="h-3.5 w-3.5 text-tks-gold" />
                      <span className="text-tks-gold">Life OS</span>
                      <span className="ml-auto text-muted-foreground">Switch to Sprint</span>
                    </>
                  )}
                </button>
              </div>
            )}
          </SidebarContent>

          <SidebarFooter className="p-3">
            {!isCollapsed && (
              <p className="text-[10px] text-muted-foreground/50 text-center italic mb-2">
                Est Ars Celare Artem
              </p>
            )}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="flex items-center gap-3 rounded-lg px-1 py-1 hover:bg-accent/50 transition-colors w-full text-left group-data-[collapsible=icon]:justify-center focus:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                  <Avatar className="h-9 w-9 border border-tks-gold/20 shrink-0">
                    <AvatarFallback className="text-xs font-medium bg-tks-gold/10 text-tks-gold">
                      {user?.name?.charAt(0).toUpperCase()}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1 min-w-0 group-data-[collapsible=icon]:hidden">
                    <p className="text-sm font-medium truncate leading-none">{user?.name || "-"}</p>
                    <p className="text-xs text-muted-foreground truncate mt-1.5">{user?.email || "-"}</p>
                  </div>
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem onClick={logout} className="cursor-pointer text-destructive focus:text-destructive">
                  <LogOut className="mr-2 h-4 w-4" />
                  <span>Sign out</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarFooter>
        </Sidebar>
        <div
          className={`absolute top-0 right-0 w-1 h-full cursor-col-resize hover:bg-tks-gold/20 transition-colors ${isCollapsed ? "hidden" : ""}`}
          onMouseDown={() => { if (!isCollapsed) setIsResizing(true); }}
          style={{ zIndex: 50 }}
        />
      </div>

      <SidebarInset>
        {isMobile && (
          <div className="flex border-b h-14 items-center justify-between bg-background/95 px-2 backdrop-blur supports-[backdrop-filter]:backdrop-blur sticky top-0 z-40">
            <div className="flex items-center gap-2">
              <SidebarTrigger className="h-9 w-9 rounded-lg bg-background" />
              <span className="tracking-tight text-foreground">{activeMenuItem?.label ?? "Menu"}</span>
            </div>
          </div>
        )}
        <main className="flex-1 p-4 md:p-6">{children}</main>
      </SidebarInset>
    </>
  );
}
