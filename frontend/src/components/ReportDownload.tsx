"use client";

import React from "react";
import { useLanguage } from "@/context/LanguageContext";
import { FileText, Download } from "lucide-react";
import { getPdfReportUrl } from "@/lib/api";

interface ReportDownloadProps {
  diagnosisName: string;
  confidence: number;
}

export default function ReportDownload({ diagnosisName, confidence }: ReportDownloadProps) {
  const { t } = useLanguage();
  const pdfUrl = getPdfReportUrl(diagnosisName, confidence);

  return (
    <div className="p-6 rounded-3xl bg-gradient-to-br from-rose-900/40 via-slate-900 to-slate-900 border border-rose-500/20 shadow-sm flex flex-col justify-between">
      <div>
        <div className="flex items-center gap-2.5 text-sm font-bold text-white mb-1">
          <FileText className="w-5 h-5 text-rose-400" />
          <span>{t("sec3_report_title")}</span>
        </div>
        <p className="text-xs text-slate-400 mb-4">{t("sec3_report_desc")}</p>
      </div>

      <div className="pt-2">
        <a
          href={pdfUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center justify-center gap-2 w-full py-2.5 px-4 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs transition-colors shadow-md hover:shadow-rose-500/20"
        >
          <Download className="w-4 h-4" />
          <span>{t("sec3_btn_download_pdf")}</span>
        </a>
      </div>
    </div>
  );
}
