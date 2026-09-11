# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
Khi temperature tăng từ 0.0 lên 1.5, phản hồi chuyển từ tính ổn định, chuẩn xác và lặp lại (ở 0.0 luôn nói về sự thật phổ biến như cà phê xuất khẩu) sang đa dạng chủ đề và giàu cảm xúc hơn (ở 0.5 - 1.0 chuyển sang Sơn Đoòng, Củ Chi). Khi lên đến 1.5, từ ngữ trở nên rất bay bổng và tiềm ẩn nguy cơ sai lệch thông tin (hallucination) cao hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
Tôi sẽ chọn temperature từ 0.0 đến 0.2. Đối với chatbot hỗ trợ khách hàng, độ chính xác và tính nhất quán là ưu tiên hàng đầu. Nhiệt độ thấp giúp giảm thiểu tối đa hiện tượng bịa đặt thông tin (hallucination), đảm bảo khách hàng luôn nhận được câu trả lời chuẩn xác về giá cả và chính sách dịch vụ.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
Cho workload này, GPT-4o đắt hơn GPT-4o-mini xấp xỉ 16.7 lần ($105.00/ngày so với $6.30/ngày cho 10.5 triệu output token, chênh lệch gần $3,000/tháng).
- Trường hợp GPT-4o xứng đáng: Khi thực hiện các tác vụ suy luận logic phức tạp, phân tích hợp đồng pháp lý, tài chính hoặc sinh mã lập trình đa tầng đòi hỏi độ chính xác cao nhất.
- Trường hợp nên dùng GPT-4o-mini: Các tác vụ thông thường tần suất cao như phân loại ý định (intent routing), trích xuất thông tin đơn giản, tóm tắt văn bản hoặc trả lời câu hỏi FAQ sẵn có.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
Hai phản hồi khác biệt rõ rệt về từ vựng, văn phong và cấu trúc: bản cho trẻ em dùng hình ảnh ẩn dụ trực quan (cuốn sổ chung của lớp học) với câu văn ngắn gọn, dễ hiểu; trong khi bản cho chuyên gia tài chính sử dụng chuẩn thuật ngữ học thuật (DLT, sổ cái phân tán, mạng ngang hàng peer-to-peer). System prompt đóng vai trò định hình sâu sắc vai diễn (persona), đối tượng người nghe và phong cách diễn đạt của mô hình mà không cần thay đổi câu hỏi gốc của người dùng.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
Số token đếm bằng tiktoken và ước lượng `số từ / 0.75` chênh lệch nhau khoảng 20% (ước lượng thô thường cao hơn số token thực tế trong đoạn văn này). Tiếng Việt thường tốn nhiều token hơn tiếng Anh cùng độ dài ngữ nghĩa vì thuật toán Byte-Pair Encoding (BPE) của tokenizer được huấn luyện chủ yếu trên kho ngữ liệu tiếng Anh, nơi các từ phổ biến thường là 1 token nguyên vẹn; trong khi tiếng Việt có dấu thanh và nhiều ký tự có dấu (Unicode đa byte), khiến tokenizer thường xuyên phải tách 1 từ tiếng Việt thành nhiều sub-tokens hoặc byte đơn lẻ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
Streaming quan trọng nhất trong các ứng dụng tương tác trực tiếp với người dùng như chatbot hoặc trợ lý ảo đối thoại thời gian thực, nơi thời gian phản hồi ban đầu (Time-to-First-Token - TTFT) quyết định cảm nhận mượt mà của người dùng và giảm bớt sự khó chịu khi phải chờ đợi phản hồi dài. Ngược lại, non-streaming phù hợp hơn cho các tác vụ xử lý ngầm (batch processing, background jobs), phân tích dữ liệu hàng loạt, sinh dữ liệu cấu trúc (JSON / Structured Outputs cần validate toàn bộ schema trước khi parse) hoặc khi ứng dụng giao tiếp giữa máy với máy (API backend-to-backend) không có người dùng trực tiếp quan sát.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
Khi API bị quá tải hoặc đạt giới hạn rate-limit, exponential backoff (thời gian chờ tăng gấp đôi sau mỗi lần thử: 1s, 2s, 4s, 8s...) giúp giãn cách dần áp lực request, cho hệ thống máy chủ đủ thời gian hồi phục và giải phóng tài nguyên. Nếu hàng nghìn client cùng retry với delay cố định giống nhau (ví dụ đều chờ đúng 1 giây), toàn bộ các request này sẽ đồng loạt ập vào server tại cùng một thời điểm sau mỗi giây, gây ra hiện tượng bão yêu cầu (thundering herd problem / retry storm), khiến máy chủ tiếp tục bị quá tải liên tục và không thể phục hồi được.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
Tôi chọn persona: "Trợ giảng AI hỗ trợ sinh viên lập trình".
System prompt: "Bạn là một trợ giảng AI thân thiện, kiên nhẫn chuyên hỗ trợ sinh viên học lập trình Python. Hãy giải thích các khái niệm kỹ thuật một cách dễ hiểu, có ví dụ minh họa ngắn và luôn trả lời bằng tiếng Việt ngắn gọn, súc tích trong tối đa 3-4 câu."
Giải thích lựa chọn từ ngữ:
1. Yêu cầu "trả lời ngắn gọn, súc tích trong tối đa 3-4 câu": Giúp kiểm soát số lượng output token sinh ra, tiết kiệm chi phí API và tránh làm người học bị ngợp trước những đoạn văn giải thích quá dài dòng trong cửa sổ dòng lệnh CLI.
2. Chỉ định "luôn trả lời bằng tiếng Việt": Đảm bảo tính nhất quán về ngôn ngữ giao tiếp, tránh trường hợp mô hình tự động chuyển sang tiếng Anh khi gặp các từ khóa lập trình kỹ thuật.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
Hạn chế lớn nhất hiện tại của trợ lý là cửa sổ ngữ cảnh ngắn (chỉ lưu 3 lượt hội thoại gần nhất = 6 message) và không có bộ nhớ dài hạn (persistent memory), khiến trợ lý quên hoàn toàn thông tin người dùng đã chia sẻ ở các lượt trước đó hoặc khi khởi động lại chương trình.
Đề xuất cải thiện: Triển khai cơ chế "Tóm tắt ngữ cảnh tự động" (Conversation Summarization) kết hợp lưu trữ file JSON hoặc cơ sở dữ liệu SQLite.
Cách triển khai: Khi lịch sử vượt quá 6 tin nhắn, thay vì cắt bỏ thẳng tay các tin nhắn cũ, ta gọi một model nhỏ (như GPT-4o-mini) để tóm tắt các tin nhắn cũ thành 1 đoạn ngắn gọn (summary) và chèn đoạn tóm tắt này vào đầu danh sách messages ngay sau system prompt; đồng thời lưu toàn bộ lịch sử và tóm tắt này xuống file local để có thể nạp lại khi người dùng mở lại phiên chat mới.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
