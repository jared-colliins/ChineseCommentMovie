-- 在 Supabase Dashboard 的 SQL Editor 中执行一次。
create table if not exists public.review_feedback (
  id bigint generated always as identity primary key,
  user_id uuid not null default auth.uid() references auth.users(id),
  text text not null check (char_length(text) between 1 and 500),
  label smallint not null check (label in (0, 1)),
  created_at timestamptz not null default now()
);

alter table public.review_feedback enable row level security;

-- 项目关闭了“自动暴露新表”，因此显式授予登录用户最小写入权限。
grant usage on schema public to authenticated;
grant insert on table public.review_feedback to authenticated;
grant usage on sequence public.review_feedback_id_seq to authenticated;

-- 已登录用户只能插入自己的反馈，不能读取其他人的内容。
create policy "authenticated users can submit feedback"
on public.review_feedback
for insert
to authenticated
with check (auth.uid() = user_id and char_length(text) between 1 and 500 and label in (0, 1));
