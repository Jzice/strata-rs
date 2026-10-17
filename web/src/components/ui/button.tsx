import { forwardRef } from "react";
import { cn } from "../../lib/cn";

type Variant = "primary" | "ghost" | "outline" | "danger" | "subtle";
type Size = "sm" | "md" | "icon";

const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-brand-gradient text-white shadow-glow hover:brightness-110 active:brightness-95 disabled:opacity-50",
  ghost: "text-gray-300 hover:bg-white/10 hover:text-white",
  outline: "border border-white/15 text-gray-200 hover:border-brand-500/60 hover:text-white",
  danger: "bg-danger/90 text-white hover:bg-danger disabled:opacity-50",
  subtle: "bg-white/[0.06] text-gray-200 hover:bg-white/[0.12]",
};

const SIZES: Record<Size, string> = {
  sm: "h-8 px-3 text-[13px] gap-1.5",
  md: "h-10 px-4 text-sm gap-2",
  icon: "h-9 w-9 justify-center",
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className, variant = "subtle", size = "md", type = "button", ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        "inline-flex items-center rounded-xl font-medium transition-all duration-200",
        "focus-visible:ring-focus cursor-pointer disabled:cursor-not-allowed active:scale-[0.98]",
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...rest}
    />
  );
});
