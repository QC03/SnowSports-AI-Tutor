import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading

# 기존 run_pipeline 및 feedback_engine 모듈에서 필요한 자원 불러오기
from feedback_engine import SPORT_CURRICULUM, build_selection, build_selection_summary
import run_pipeline

class PipelineApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("스포츠 포즈 분석 시스템 (Sports Pose Analyzer)")
        self.root.geometry("700x650")
        self.root.minsize(600, 500)

        # 변수 선언
        self.demo_video_path = tk.StringVar()
        self.user_video_path = tk.StringVar()
        self.output_dir_path = tk.StringVar(value="analysis_outputs")
        self.threshold_degrees = tk.DoubleVar(value=15.0)
        self.frame_step = tk.IntVar(value=1)

        # UI 컴포넌트 배치
        self._create_widgets()
        self._bind_combobox_events()

    def _create_widgets(self):
        # 1. 파일 선택 프레임
        file_frame = ttk.LabelFrame(self.root, text="비디오 및 출력 경로 설정", padding=10)
        file_frame.pack(fill="x", padx=15, pady=10)

        # 데모 비디오
        ttk.Label(file_frame, text="데모 비디오:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(file_frame, textvariable=self.demo_video_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="찾아보기", command=self._browse_demo).grid(row=0, column=2, padx=5)

        # 사용자 비디오
        ttk.Label(file_frame, text="사용자 비디오:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(file_frame, textvariable=self.user_video_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="찾아보기", command=self._browse_user).grid(row=1, column=2, padx=5)

        # 출력 디렉토리
        ttk.Label(file_frame, text="출력 디렉토리:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(file_frame, textvariable=self.output_dir_path, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(file_frame, text="찾아보기", command=self._browse_output).grid(row=2, column=2, padx=5)

        # 2. 스포츠 및 커리큘럼 선택 프레임
        curr_frame = ttk.LabelFrame(self.root, text="스포츠 / 기술 선택", padding=10)
        curr_frame.pack(fill="x", padx=15, pady=10)

        ttk.Label(curr_frame, text="종목 (Sport):").grid(row=0, column=0, sticky="w", pady=5)
        self.sport_combo = ttk.Combobox(curr_frame, values=sorted(SPORT_CURRICULUM.keys()), state="readonly", width=30)
        self.sport_combo.grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(curr_frame, text="레벨 (Level):").grid(row=1, column=0, sticky="w", pady=5)
        self.level_combo = ttk.Combobox(curr_frame, state="readonly", width=30)
        self.level_combo.grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(curr_frame, text="기술 (Technique):").grid(row=2, column=0, sticky="w", pady=5)
        self.tech_combo = ttk.Combobox(curr_frame, state="readonly", width=30)
        self.tech_combo.grid(row=2, column=1, sticky="w", padx=5)

        # 3. 상세 파라미터 설정 프레임
        param_frame = ttk.LabelFrame(self.root, text="분석 파라미터 설정", padding=10)
        param_frame.pack(fill="x", padx=15, pady=5)

        ttk.Label(param_frame, text="프레임 스킵 간격 (Frame Step):").grid(row=0, column=2, sticky="w", padx=5)
        ttk.Entry(param_frame, textvariable=self.frame_step, width=10).grid(row=0, column=3, sticky="w", padx=5)

        # 4. 분석 실행 및 로그창
        self.run_btn = ttk.Button(self.root, text="포즈 분석 파이프라인 실행", command=self._start_pipeline_thread, style="Accent.TButton")
        self.run_btn.pack(pady=10)

        log_frame = ttk.LabelFrame(self.root, text="실행 로그 및 피드백 보고서", padding=10)
        log_frame.pack(fill="both", expand=True, padx=15, pady=10)

        self.log_text = tk.Text(log_frame, wrap="word", height=15)
        self.log_text.pack(side="left", fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

    def _bind_combobox_events(self):
        # 콤보박스 선택 상태에 따라 하위 리스트를 유기적으로 변경
        self.sport_combo.bind("<<ComboboxSelected>>", self._on_sport_selected)
        self.level_combo.bind("<<ComboboxSelected>>", self._on_level_selected)

    def _on_sport_selected(self, event):
        sport = self.sport_combo.get()
        if sport:
            levels = sorted(SPORT_CURRICULUM[sport].keys())
            self.level_combo.config(values=levels)
            self.level_combo.set("")
            self.tech_combo.config(values=[])
            self.tech_combo.set("")

    def _on_level_selected(self, event):
        sport = self.sport_combo.get()
        level = self.level_combo.get()
        if sport and level:
            techniques = sorted(SPORT_CURRICULUM[sport][level])
            self.tech_combo.config(values=techniques)
            self.tech_combo.set("")

    # 경로 브라우징 함수들
    def _browse_demo(self):
        file = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if file: self.demo_video_path.set(file)

    def _browse_user(self):
        file = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if file: self.user_video_path.set(file)

    def _browse_output(self):
        directory = filedialog.askdirectory()
        if directory: self.output_dir_path.set(directory)

    def _log(self, message: str):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def _clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def _start_pipeline_thread(self):
        # GUI 얼어붙음(동결) 방지를 위해 멀티스레딩으로 백그라운드 파이프라인 처리
        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def _run_pipeline(self):
        # 유효성 검사
        demo = self.demo_video_path.get()
        user = self.user_video_path.get()
        sport = self.sport_combo.get()
        level = self.level_combo.get()
        technique = self.tech_combo.get()

        if not demo or not user:
            messagebox.showerror("오류", "데모 영상과 사용자 영상 경로를 모두 지정해주세요.")
            return
        if not sport or not level or not technique:
            messagebox.showerror("오류", "스포츠 종목, 레벨, 기술을 모두 선택해주세요.")
            return

        # 버튼 비활성화 및 로그 초기화
        self.run_btn.config(state="disabled")
        self._clear_log()
        self._log("==== 포즈 분석 파이프라인 구동 시작 ====")
        
        # sys.stdout을 GUI 로그창으로 임시 리다이렉트하여 print 문 실시간 표시
        class StdoutRedirector:
            def __init__(self, log_func): self.log_func = log_func
            def write(self, string): 
                if string.strip(): self.log_func(string.strip())
            def flush(self): pass

        old_stdout = sys.stdout
        sys.stdout = StdoutRedirector(self._log) # type: ignore

        try:
            # 1. 아규먼트 Mocking 객체 생성
            class MockArgs:
                def __init__(self, d, u, o, s, l, t, th, f):
                    self.demo_video = d
                    self.user_video = u
                    self.output_dir = o
                    self.sport = s
                    self.level = l
                    self.technique = t
                    self.event = None  # UI 선택을 우선순위로 둠
                    self.threshold_degrees = th
                    self.frame_step = f

            args = MockArgs(
                demo, user, self.output_dir_path.get(), 
                sport, level, technique, 
                self.threshold_degrees.get(), self.frame_step.get()
            )

            # 2. 내부 run_pipeline 메커니즘을 그대로 복사/UI 연동형으로 구동
            selection = build_selection(sport, level, technique)
            demo_video_path = Path(args.demo_video)
            user_video_path = Path(args.user_video)
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            demo_pose_json = run_pipeline._video_pose_output_path(demo_video_path, output_dir)
            user_pose_json = run_pipeline._video_pose_output_path(user_video_path, output_dir)

            # [핵심] OpenCV ROI 선택 팝업창 연동을 위한 안내 메시지 팝업창
            self._log("\n⚠️ [안내] 팝업창이 뜨면 영상의 마우스 ROI 크롭을 진행해 주세요.")
            messagebox.showinfo("ROI 지정 안내", "확인을 누르시면 영상 팝업창이 뜹니다.\n추적할 대상을 드래그 후 Space 또는 Enter를 치세요.")
            
            from multiprocessing import Process
            procs = []
            procs.append(Process(target=run_pipeline._run_extraction, args=(demo_video_path, demo_pose_json, args.frame_step)))
            procs.append(Process(target=run_pipeline._run_extraction, args=(user_video_path, user_pose_json, args.frame_step)))
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join()

            self._log("\n포즈 추출이 완료되었습니다. 데이터 분석 및 동기화를 진행합니다...")
            demo_frames = run_pipeline._load_pose_json(demo_pose_json)
            user_frames = run_pipeline._load_pose_json(user_pose_json)

            demo_sequence = run_pipeline._analyze_pose_sequence(demo_frames)
            user_sequence = run_pipeline._analyze_pose_sequence(user_frames)

            sync_result = run_pipeline.synchronize_angle_sequences(
                demo_sequence["outside_knee_angles"],
                user_sequence["outside_knee_angles"],
                threshold_degrees=args.threshold_degrees,
            )

            analysis_result = run_pipeline._build_analysis_result(demo_sequence, user_sequence, sync_result)
            sync_summary = run_pipeline._build_sync_summary(sync_result)
            feedback_text = run_pipeline.generate_llm_feedback_report(selection, analysis_result, sync_summary)
            
            # 파일 저장
            import json
            summary_path = output_dir / "analysis_summary.json"
            with summary_path.open("w", encoding="utf-8") as file_handle:
                json.dump({
                    "selection": build_selection_summary(selection),
                    "distance": sync_result.distance,
                    "path": sync_result.path,
                    "anomaly_frames": sync_result.anomaly_frames,
                    "analysis_result": analysis_result,
                    "sync_summary": sync_summary,
                    "llm_feedback": feedback_text,
                }, file_handle, ensure_ascii=False, indent=2)

            self._log(f"\n==== 분석 완료 ====")
            self._log(f"결과 저장 완료: {summary_path}")
            self._log(f"선택 기술: {selection.label}")
            self._log(f"매칭된 프레임 수: {len(sync_result.path)} / 이상 프레임 수: {len(sync_result.anomaly_frames)}")
            self._log(f"\n[AI 피드백 보고서]\n{feedback_text}")
            
            messagebox.showinfo("성공", "포즈 동기화 및 LLM 피드백 리포트 생성이 완료되었습니다!")

        except Exception as e:
            self._log(f"\n❌ 에러 발생: {str(e)}")
            messagebox.showerror("오류", f"파이프라인 실행 중 오류가 발생했습니다:\n{str(e)}")
        finally:
            sys.stdout = old_stdout
            self.run_btn.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = PipelineApp(root)
    root.mainloop()