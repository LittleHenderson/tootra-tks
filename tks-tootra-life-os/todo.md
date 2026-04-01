# TKS Tootra Life OS — Project TODO

## Database & Schema
- [x] Design and push tasks table with full TKS fields
- [x] Design and push power_inventory table for 8 resource categories
- [x] Design and push dependencies table for task relationships
- [x] Design and push review_snapshots table for weekly reviews
- [x] User mode field added to users table

## Shared Types & Constants
- [x] Create TKS taxonomy constants (7 Foundations, 28 Sub-Foundations, 4 Worlds, 10 Noetics, 11 Operators)
- [x] Create D/W/P cycle definitions (12 stages with card/zodiac associations)
- [x] Create task archetype definitions (6 types)
- [x] Create TKS equation token type definitions (13 types with colors/icons)
- [x] Create priority scoring weights and formulas (6 factors summing to 1.0)
- [x] Create power categories (8 resource types)
- [x] Create operational modes (life_os, work_sprint)

## Server / API
- [x] Task CRUD tRPC router (create, list, update, delete, status transitions)
- [x] AI-powered task interpretation (Foundation mapping + equation generation + archetype detection)
- [x] D/W/P lifecycle tracking procedures
- [x] Power inventory CRUD procedures
- [x] Foundation heatmap analytics procedure
- [x] Priority scoring calculation (AI-powered)
- [x] Weekly review analytics procedure
- [x] CSV/JSON export procedure
- [x] User settings (mode toggle) procedure
- [x] AI subtask generation for power gaps

## Frontend — Global
- [x] Brand-consistent theming (gold/blue/flame, dark theme)
- [x] Dashboard layout with sidebar navigation (7 items)
- [x] Mode toggle (Life OS / Work Sprint)
- [x] Motto "Est Ars Celare Artem" integration
- [x] Triangle motif throughout
- [x] TKS-branded sign-in page

## Frontend — Task Management
- [x] Task capture form with AI interpretation
- [x] Task inbox view with status machine (inbox→scheduled→active→done/dropped)
- [x] Task detail view with equation display and D/W/P tracking
- [x] Task archetype badges (Execution, Prep, Maintenance, Repair, Study, Audit)
- [x] Priority-sorted task list with scoring breakdown

## Frontend — TKS Equation Visualization
- [x] Color-coded chip display for 13 token types
- [x] Equation row renderer with operator display
- [x] Canonical string form display
- [x] Token legend on Equations page

## Frontend — Foundation Heatmap
- [x] 7-cell interactive foundation heatmap
- [x] Drill-down to 28 sub-foundations
- [x] Overload and neglect alerts
- [x] Balance analytics display with deviation

## Frontend — D/W/P Lifecycle
- [x] 12-stage D/W/P tracker visualization
- [x] Mismatch detection display with alerts
- [x] Diagnostic engine results from AI
- [x] Inline stage editors
- [x] Aggregate distribution charts

## Frontend — Power Gap Analysis
- [x] 8-category power inventory checklist
- [x] Gap visualization per task (have/lack/partial)
- [x] Auto-generated subtask suggestions for gaps

## Frontend — Weekly Review Dashboard
- [x] Foundation balance trends
- [x] Completion rates display
- [x] Drift analysis
- [x] CSV and JSON export buttons
- [x] Review snapshot saving and history

## Frontend — Work Sprint Mode
- [x] Minimal overhead task view (only Inbox + Tasks visible)
- [x] Mode toggle in sidebar
- [x] Hidden TKS details in sprint mode

## Enhancements
- [x] Deterministic server-side priority scoring engine (6 weighted factors)
- [x] Recalculate priority endpoint on router
- [x] Improved export flow using tRPC client hooks
- [x] Work Sprint quick-capture bar (single-line input, Enter to capture)
- [x] Sprint mode simplified task actions (one-click Start/Done)
- [x] Sprint mode hides TKS metadata for reduced overhead

## Testing
- [x] Auth logout test
- [x] TKS shared types validation (38 tests covering all domain entities)
- [x] Auth.me and settings.getMode router tests
- [x] Priority scoring engine tests (21 tests covering all 6 scorers + composite)

## Voice-to-Text Capture
- [x] Server-side tRPC endpoint to receive audio and transcribe via Whisper
- [x] VoiceRecorder React component with mic button, recording state, and waveform indicator
- [x] Integrate voice capture into Inbox page (Home.tsx) capture area
- [x] Integrate voice capture into Tasks page quick-capture bar
- [x] Handle browser permissions, errors, and unsupported browsers gracefully
- [x] Write vitest tests for the transcription endpoint

## Text-to-Speech (Read Aloud)
- [x] ReadAloud React component with play/pause/stop controls and voice selection
- [x] Integrate Read Aloud into Task Detail page (title, description, D/W/P diagnosis)
- [x] Integrate Read Aloud into Inbox interpretation result
- [x] Integrate Read Aloud into D/W/P Tracker page
- [x] Write vitest tests for TTS component logic

## In-App Onboarding Slideshow
- [x] OnboardingSlideshow component with multi-step slides, progress dots, next/prev/skip controls
- [x] Slide 1: Welcome — what is TKS Tootra and why it matters
- [x] Slide 2: Capture — how to type or speak a task and what happens (AI mapping)
- [x] Slide 3: Foundations — the 7 Foundations explained simply with color-coded visuals
- [x] Slide 4: D/W/P Lifecycle — Desire, Wisdom, Power stages explained
- [x] Slide 5: TKS Equations — what the color-coded chips mean
- [x] Slide 6: Task Management — the status flow (inbox → scheduled → active → done)
- [x] Slide 7: Heatmap & Power Gaps — how to spot imbalances and close gaps
- [x] Slide 8: Weekly Review — how to track progress and export data
- [x] Slide 9: Modes — Life OS vs Work Sprint explained
- [x] Slide 10: Voice & Read Aloud — speak tasks and listen to summaries
- [x] Slide 11: Get Started — CTA to capture first task
- [x] Auto-show on first visit, dismissible, accessible from sidebar "How to Use" link
- [x] Persist "seen" state in localStorage so it doesn't repeat
- [x] Add "How to Use" nav item in sidebar to relaunch slideshow anytime

## Scenarioize Vision Feature
- [x] Server-side photo upload endpoint (S3 storage for user profile photo)
- [x] Database field for user's uploaded photo URL
- [x] Server-side scenarioize endpoint using AI image generation with task context
- [x] Generate single realistic AI scenario image (default mode)
- [x] Generate dual vision: "If you do this" (positive) vs "If you don't" (negative)
- [x] User can choose single image or dual comparison
- [x] Scenarioize React component with photo upload, dual-vision display, loading states
- [x] "Scenarioize" button on TaskDetail page
- [x] Integration with TKS context (Foundation, D/W/P stage, archetype, description)
- [x] Vitest tests for scenario router (11 tests)
- [x] Await all async expectations in scenario tests to fix Vitest warnings

## PWA (Progressive Web App) Support
- [x] Web app manifest (manifest.json) with TKS branding, icons, theme colors
- [x] Service worker for offline caching and background sync
- [x] Install prompt banner/button for "Add to Home Screen" on mobile and desktop
- [x] PWA-optimized icons (192x192, 512x512) with TKS triangle motif
- [x] Meta tags for iOS Safari (apple-touch-icon, status-bar-style, etc.)
- [x] Splash screen configuration for iOS and Android
- [x] Offline fallback page
- [x] Update index.html with PWA meta tags and manifest link
- [x] PWA vitest tests (22 tests covering manifest, SW, offline page, index.html)

## React Native / Expo Wrapper
- [x] Initialize Expo project with TypeScript
- [x] TKS brand theme system (gold/blue/flame colors, Inter font, dark mode)
- [x] Bottom tab navigation with TKS-branded icons (Inbox, Tasks, Heatmap, Profile)
- [x] Inbox screen with voice capture and task creation
- [x] Tasks screen with status filtering and search
- [x] Task Detail screen with Equation chips, D/W/P timeline, Scenarioize button
- [x] Heatmap screen with Foundation visualization and drill-down
- [x] Profile screen with mode toggle (Life OS / Work Sprint), settings, menu
- [x] Reusable components: EquationChip, FoundationBadge, DWPBadge, TaskCard, TriangleLogo
- [x] API client connecting to the web app's tRPC backend
- [x] Equations screen with token legend and chip display
- [x] Power Gaps screen with 8-category inventory and gap visualization
- [x] Weekly Review screen with trends, completion rates, drift analysis, and export
- [x] D/W/P Tracker screen with distribution chart, filters, and mismatch diagnostics
- [x] Onboarding slideshow (11 slides) with first-launch detection via AsyncStorage
- [x] Profile screen navigates to all feature screens (Equations, PowerGaps, WeeklyReview, DWPTracker)
- [x] Comprehensive README with setup instructions
- [x] Push to GitHub repo (LittleHenderson/tks-tootra-mobile)

## Scenarioize Discoverability Improvements
- [x] Add dedicated Scenarioize slide (#11) to the onboarding slideshow (now 12 slides)
- [x] Add Scenarioize quick-action button on Inbox task cards (Home.tsx)
- [x] Scenarioize auto-opens on TaskDetail when navigated from Inbox quick-action
- [x] Scenarioize component supports autoOpen prop
- [x] Get Started slide checklist updated to include Scenarioize step
- [x] Add Scenarioize quick-action button on Tasks page task cards

## Live Walkthrough Screenshots in Onboarding
- [x] Capture screenshot: Inbox page with capture input
- [x] Capture screenshot: Task captured with AI interpretation result
- [x] Capture screenshot: Task Detail page with equation, D/W/P, power inventory
- [x] Capture screenshot: Tasks page with status filters and Scenarioize buttons
- [x] Capture screenshot: Heatmap page with foundation visualization
- [x] Capture screenshot: Equations page with token display
- [x] Capture screenshot: Power Gaps page
- [x] Capture screenshot: Weekly Review dashboard
- [x] Capture screenshot: D/W/P Tracker page
- [x] Capture screenshot: Scenarioize panel open with photo upload
- [x] Upload all screenshots to CDN via manus-upload-file --webdev
- [x] Integrate real screenshots into onboarding slideshow slides
- [x] Each slide shows the real UI alongside the explanation text

## Onboarding Skip Button Enhancement
- [x] Add prominent "Skip" button on onboarding slideshow for users who want to explore directly
- [x] Ensure skip button is visible on every slide (not just in footer)

## GitHub Push
- [ ] Push latest code to LittleHenderson/tootra-tks repo

## Interactive Tour Mode
- [ ] Build TourMode component with highlight overlay + tooltip on actual UI elements
- [ ] Tour Step 1: Highlight the task capture input area with tooltip "Type your task here"
- [ ] Tour Step 2: Highlight the Capture & Map button with tooltip "Click to let AI analyze your task"
- [ ] Tour Step 3: Highlight the Inbox section with tooltip "Your captured tasks appear here"
- [ ] Tour Step 4: Highlight the sidebar navigation with tooltip "Explore all TKS features from here"
- [ ] Tour Step 5: Highlight the Scenarioize button with tooltip "Visualize your future outcomes"
- [ ] Tour Step 6: Highlight the mode toggle with tooltip "Switch between Life OS and Work Sprint"
- [ ] Add "Start Tour" option at end of onboarding slideshow
- [ ] Add "Take a Tour" button accessible from sidebar or How to Use page
- [ ] Tour auto-advances when user completes each highlighted action
- [ ] Tour can be dismissed at any step
- [ ] Persist tour completion state in localStorage
- [ ] Write vitest tests for Tour Mode component logic
