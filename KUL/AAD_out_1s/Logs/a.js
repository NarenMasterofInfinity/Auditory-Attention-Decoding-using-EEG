"use strict";
(self.webpackChunkfrontend = self.webpackChunkfrontend || []).push([
    [792], {
        1676: (z, W, m) => {
            var F = m(467),
                D = m(7705),
                s = m(2615),
                J = m(5635),
                M = m(7916),
                y = m(5312),
                S = m(4988),
                R = m(1943),
                d = m(3955),
                f = m(2629),
                B = m(455),
                b = m(8834),
                t = m(3664),
                w = m(1585),
                $ = m(7827);

            function Ve(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 4), t.nI1(1, "translate"), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.startHackingInstructor())
                    }), t.j41(2, "mat-icon"), t.EFF(3, " school "), t.k0s(), t.j41(4, "span", 5), t.EFF(5), t.nI1(6, "translate"), t.k0s()()
                }
                2 & n && (t.Y8G("matTooltip", t.bMT(1, 2, "SCORE_BOARD_HACKING_INSTRUCTOR")), t.R7$(5), t.JRh(t.bMT(6, 4, "BTN_GETTING_STARTED")))
            }

            function Ge(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 6), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.closeWelcome())
                    }), t.j41(1, "mat-icon"), t.EFF(2, " visibility_off "), t.k0s(), t.j41(3, "span", 7), t.EFF(4), t.nI1(5, "translate"), t.k0s()()
                }
                2 & n && (t.R7$(4), t.JRh(t.bMT(5, 1, "BTN_DISMISS")))
            }
            let Ye = (() => {
                    class n {
                        dialogRef;
                        configurationService;
                        cookieService;
                        title = "Welcome to OWASP Juice Shop";
                        message = "<p>Being a web application with a vast number of intended security vulnerabilities, the <strong>OWASP Juice Shop</strong> is supposed to be the opposite of a best practice or template application for web developers: It is an awareness, training, demonstration and exercise tool for security risks in modern web applications. The <strong>OWASP Juice Shop</strong> is an open-source project hosted by the non-profit <a href='https://owasp.org' target='_blank'>Open Worldwide Application Security Project (OWASP)</a> and is developed and maintained by volunteers. Check out the link below for more information and documentation on the project.</p><h1><a href='https://owasp-juice.shop' target='_blank'>https://owasp-juice.shop</a></h1>";
                        showHackingInstructor = !0;
                        showDismissBtn = !0;
                        welcomeBannerStatusCookieKey = "welcomebanner_status";
                        constructor(e, o, i) {
                            this.dialogRef = e, this.configurationService = o, this.cookieService = i
                        }
                        ngOnInit() {
                            this.configurationService.getApplicationConfiguration().subscribe({
                                next: e => {
                                    e?.application?.welcomeBanner && (this.title = e.application.welcomeBanner.title, this.message = e.application.welcomeBanner.message), this.showHackingInstructor = e?.hackingInstructor?.isEnabled, this.showHackingInstructor && e?.challenges?.restrictToTutorialsFirst && (this.dialogRef.disableClose = !0, this.showDismissBtn = !1)
                                },
                                error: e => {
                                    console.log(e)
                                }
                            })
                        }
                        startHackingInstructor() {
                            this.closeWelcome(), console.log('Starting instructions for challenge "Score Board"'), m.e(781).then(m.bind(m, 1143)).then(e => {
                                e.startHackingInstructorFor("Score Board")
                            })
                        }
                        closeWelcome() {
                            this.dialogRef.close();
                            const e = new Date;
                            e.setFullYear(e.getFullYear() + 1), this.cookieService.put(this.welcomeBannerStatusCookieKey, "dismiss", {
                                expires: e
                            })
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(w.CP), t.rXU(M.s), t.rXU($.O1))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-welcome-banner"]
                            ],
                            decls: 6,
                            vars: 4,
                            consts: [
                                [1, "text-justify", 3, "innerHtml"],
                                [1, "button-footer"],
                                ["mat-raised-button", "", "color", "warn", "matTooltipPosition", "above", 3, "matTooltip"],
                                ["mat-raised-button", "", "color", "primary", "aria-label", "Close Welcome Banner", 1, "close-dialog"],
                                ["mat-raised-button", "", "color", "warn", "matTooltipPosition", "above", 3, "click", "matTooltip"],
                                [1, "hide-lt-lg"],
                                ["mat-raised-button", "", "color", "primary", "aria-label", "Close Welcome Banner", 1, "close-dialog", 3, "click"],
                                [1, "hide-lt-sm"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "h1"), t.EFF(1), t.k0s(), t.nrm(2, "div", 0), t.j41(3, "div", 1), t.nVh(4, Ve, 7, 6, "button", 2), t.nVh(5, Ge, 6, 3, "button", 3), t.k0s()), 2 & o && (t.R7$(), t.JRh(i.title), t.R7$(), t.Y8G("innerHtml", i.message, t.npT), t.R7$(2), t.vxM(i.showHackingInstructor ? 4 : -1), t.R7$(), t.vxM(i.showDismissBtn ? 5 : -1))
                            },
                            dependencies: [b.Hl, b.$z, B.oV, f.m_, f.An, d.h, d.D9],
                            styles: ["\n\n\n\n[_nghost-%COMP%]{display:block;margin:24px}.text-justify[_ngcontent-%COMP%]{text-align:justify;text-justify:inter-word}[_nghost-%COMP%]   h1[_ngcontent-%COMP%]   a[_ngcontent-%COMP%]{font-size:20px}[_nghost-%COMP%]     strong{font-style:italic}.button-footer[_ngcontent-%COMP%]{align-items:center;display:flex;justify-content:space-between}.close-dialog[_ngcontent-%COMP%]{cursor:pointer;font-size:14px}mat-icon[_ngcontent-%COMP%]{align-items:center;font-size:24px!important;height:24px!important;justify-content:center;margin-right:1px!important;width:24px!important}@media screen and (max-width: 1279px){mat-icon[_ngcontent-%COMP%]{margin-left:1px!important}}.hide-lt-lg[_ngcontent-%COMP%]{display:inline}@media (max-width: 1279.98px){.hide-lt-lg[_ngcontent-%COMP%]{display:none!important}}.hide-lt-sm[_ngcontent-%COMP%]{display:inline}@media (max-width: 599.98px){.hide-lt-sm[_ngcontent-%COMP%]{display:none!important}}"]
                        })
                    }
                    return n
                })(),
                He = (() => {
                    class n {
                        dialog;
                        configurationService;
                        cookieService;
                        welcomeBannerStatusCookieKey = "welcomebanner_status";
                        constructor(e, o, i) {
                            this.dialog = e, this.configurationService = o, this.cookieService = i
                        }
                        ngOnInit() {
                            "dismiss" !== this.cookieService.get(this.welcomeBannerStatusCookieKey) && this.configurationService.getApplicationConfiguration().subscribe({
                                next: o => {
                                    o?.application?.welcomeBanner && !o.application.welcomeBanner.showOnFirstStart || this.dialog.open(Ye, {
                                        minWidth: "320px",
                                        width: "35%",
                                        position: {
                                            top: "50px"
                                        }
                                    })
                                },
                                error: o => {
                                    console.log(o)
                                }
                            })
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(w.bZ), t.rXU(M.s), t.rXU($.O1))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-welcome"]
                            ],
                            decls: 1,
                            vars: 0,
                            template: function(o, i) {
                                1 & o && t.Hgh(0, "div")
                            },
                            styles: ["\n\n\n\n.welcome-style[_ngcontent-%COMP%]   .mat-mdc-dialog-container[_ngcontent-%COMP%]{margin-left:auto;margin-right:auto;max-width:50%;min-width:320px}"]
                        })
                    }
                    return n
                })();
            var v = m(6354),
                g = m(9437),
                E = m(9330);
            let Z = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/api/Challenges";
                        constructor(e) {
                            this.http = e
                        }
                        find(e) {
                            return this.http.get(this.host + "/", {
                                params: e
                            }).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        repeatNotification(e) {
                            return this.http.get(this.hostServer + "/rest/repeat-notification", {
                                params: {
                                    challenge: e
                                },
                                responseType: "text"
                            }).pipe((0, g.W)(o => {
                                throw o
                            }))
                        }
                        continueCode() {
                            return this.http.get(this.hostServer + "/rest/continue-code").pipe((0, v.T)(e => e.continueCode), (0, g.W)(e => {
                                throw e
                            }))
                        }
                        continueCodeFindIt() {
                            return this.http.get(this.hostServer + "/rest/continue-code-findIt").pipe((0, v.T)(e => e.continueCode), (0, g.W)(e => {
                                throw e
                            }))
                        }
                        continueCodeFixIt() {
                            return this.http.get(this.hostServer + "/rest/continue-code-fixIt").pipe((0, v.T)(e => e.continueCode), (0, g.W)(e => {
                                throw e
                            }))
                        }
                        restoreProgress(e) {
                            return this.http.put(this.hostServer + "/rest/continue-code/apply/" + e, {}).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        restoreProgressFindIt(e) {
                            return this.http.put(this.hostServer + "/rest/continue-code-findIt/apply/" + e, {}).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        restoreProgressFixIt(e) {
                            return this.http.put(this.hostServer + "/rest/continue-code-fixIt/apply/" + e, {}).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })(),
                Xe = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        constructor(e) {
                            this.http = e
                        }
                        getCountryMapping() {
                            return this.http.get(this.hostServer + "/rest/country-mapping").pipe((0, g.W)(e => {
                                throw e
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })();
            var te = m(4382);
            let et = (() => {
                class n {
                    ngZone;
                    _socket;
                    constructor(e) {
                        this.ngZone = e, this.ngZone.runOutsideAngular(() => {
                            this._socket = "." === y.c.hostServer ? (0, te.io)(window.location.origin, {
                                path: (window.location.pathname.endsWith("/") ? window.location.pathname : window.location.pathname + "/") + "socket.io"
                            }) : (0, te.io)(y.c.hostServer)
                        })
                    }
                    socket() {
                        return this._socket
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(s.KVO(t.SKi))
                    };
                    static \u0275prov = s.jDH({
                        token: n,
                        factory: n.\u0275fac,
                        providedIn: "root"
                    })
                }
                return n
            })();
            var Pt = m(9692),
                h = m(5596),
                Y = m(2200),
                ee = m(4843);
            const We = (n, a) => a.key;

            function ze(n, a) {
                if (1 & n && (t.nrm(0, "span"), t.nI1(1, "lowercase")), 2 & n) {
                    const e = t.XpG(3).$implicit;
                    t.HbH(t.VkB("fi fi-", t.bMT(1, 3, e.country.code)))
                }
            }

            function Je(n, a) {
                1 & n && (t.j41(0, "mat-icon"), t.EFF(1, "my_location"), t.k0s())
            }

            function Qe(n, a) {
                if (1 & n && (t.j41(0, "span"), t.EFF(1), t.k0s()), 2 & n) {
                    const e = t.XpG(3).$implicit;
                    t.R7$(), t.JRh(e.country.name)
                }
            }

            function qe(n, a) {
                if (1 & n && (t.j41(0, "span", 5), t.nVh(1, ze, 2, 5, "span", 8), t.nVh(2, Je, 2, 0, "mat-icon"), t.nVh(3, Qe, 2, 1, "span"), t.k0s()), 2 & n) {
                    const e = t.XpG(3);
                    t.R7$(), t.vxM("flag" === e.showCtfCountryDetailsInNotifications || "both" === e.showCtfCountryDetailsInNotifications ? 1 : -1), t.R7$(), t.vxM("name" === e.showCtfCountryDetailsInNotifications ? 2 : -1), t.R7$(), t.vxM("name" === e.showCtfCountryDetailsInNotifications || "both" === e.showCtfCountryDetailsInNotifications ? 3 : -1)
                }
            }

            function Ke(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.nrm(0, "br"), t.j41(1, "div")(2, "span", 5)(3, "mat-icon"), t.EFF(4, "outlined_flag"), t.k0s(), t.EFF(5), t.k0s(), t.EFF(6, "\xa0 "), t.j41(7, "button", 6), t.bIt("cbOnSuccess", function() {
                        s.eBV(e);
                        const i = t.XpG().$implicit;
                        return s.Njj(i.copied = !0)
                    }), t.j41(8, "mat-icon"), t.EFF(9, "content_copy"), t.k0s(), t.j41(10, "span", 7), t.EFF(11), t.nI1(12, "translate"), t.k0s(), t.j41(13, "span", 7), t.EFF(14), t.nI1(15, "translate"), t.k0s()(), t.nrm(16, "br"), t.nVh(17, qe, 4, 3, "span", 5), t.k0s()
                }
                if (2 & n) {
                    const e = t.XpG().$implicit,
                        o = t.XpG();
                    t.R7$(5), t.SpI(" ", e.flag), t.R7$(2), t.Y8G("cbContent", e.flag)("disabled", e.copied), t.R7$(3), t.Y8G("hidden", !e.copied), t.R7$(), t.JRh(t.bMT(12, 8, "COPY_SUCCESS")), t.R7$(2), t.Y8G("hidden", e.copied), t.R7$(), t.JRh(t.bMT(15, 10, "COPY_TO_CLIPBOARD")), t.R7$(3), t.vxM("none" !== o.showCtfCountryDetailsInNotifications ? 17 : -1)
                }
            }

            function Ze(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-card", 1)(1, "div", 2)(2, "div", 3), t.EFF(3), t.j41(4, "button", 4), t.bIt("click", function(i) {
                        const r = s.eBV(e).$index,
                            c = t.XpG();
                        return s.Njj(c.closeNotification(r, i.shiftKey))
                    }), t.EFF(5, "X"), t.k0s()(), t.nVh(6, Ke, 18, 12), t.k0s()()
                }
                if (2 & n) {
                    const e = a.$implicit,
                        o = t.XpG();
                    t.R7$(3), t.JRh(e.message), t.R7$(3), t.vxM(o.showCtfFlagsInNotifications ? 6 : -1)
                }
            }
            let tn = (() => {
                class n {
                    ngZone;
                    configurationService;
                    challengeService;
                    countryMappingService;
                    translate;
                    cookieService;
                    ref;
                    io;
                    notifications = [];
                    showCtfFlagsInNotifications = !1;
                    showCtfCountryDetailsInNotifications = "none";
                    countryMap;
                    constructor(e, o, i, r, c, p, k, x) {
                        this.ngZone = e, this.configurationService = o, this.challengeService = i, this.countryMappingService = r, this.translate = c, this.cookieService = p, this.ref = k, this.io = x
                    }
                    ngOnInit() {
                        this.ngZone.runOutsideAngular(() => {
                            this.io.socket().on("challenge solved", e => {
                                e?.challenge && (e.hidden || this.showNotification(e), e.isRestore || (this.saveProgress(), e.hidden || m.e(705).then(m.bind(m, 4705)).then(o => {
                                    o.shootConfetti()
                                })), this.io.socket().emit("notification received", e.flag))
                            })
                        }), this.configurationService.getApplicationConfiguration().subscribe(e => {
                            e?.ctf && (this.showCtfFlagsInNotifications = !!e.ctf.showFlagsInNotifications && e.ctf.showFlagsInNotifications, e.ctf.showCountryDetailsInNotifications ? (this.showCtfCountryDetailsInNotifications = e.ctf.showCountryDetailsInNotifications, "none" !== e.ctf.showCountryDetailsInNotifications && this.countryMappingService.getCountryMapping().subscribe({
                                next: o => {
                                    this.countryMap = o
                                },
                                error: o => {
                                    console.log(o)
                                }
                            })) : this.showCtfCountryDetailsInNotifications = "none")
                        })
                    }
                    closeNotification(e, o = !1) {
                        o ? (this.ngZone.runOutsideAngular(() => {
                            this.io.socket().emit("verifyCloseNotificationsChallenge", this.notifications)
                        }), this.notifications = []) : this.notifications.splice(e, 1), this.ref.detectChanges()
                    }
                    showNotification(e) {
                        (0, ee._)(this.translate.get("CHALLENGE_SOLVED", {
                            challenge: e.challenge
                        })).then(o => {
                            let i;
                            this.showCtfCountryDetailsInNotifications && "none" !== this.showCtfCountryDetailsInNotifications && (i = this.countryMap[e.key]), this.notifications.push({
                                message: o,
                                key: e.key,
                                flag: e.flag,
                                country: i,
                                copied: !1
                            }), this.ref.detectChanges()
                        })
                    }
                    saveProgress() {
                        this.challengeService.continueCode().subscribe({
                            next: e => {
                                if (!e) throw new Error("Received invalid continue code from the server!");
                                const o = new Date;
                                o.setFullYear(o.getFullYear() + 1), this.cookieService.put("continueCode", e, {
                                    expires: o
                                })
                            },
                            error: e => {
                                console.log(e)
                            }
                        })
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(t.SKi), t.rXU(M.s), t.rXU(Z), t.rXU(Xe), t.rXU(d.c$), t.rXU($.O1), t.rXU(D.gRc), t.rXU(et))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-challenge-solved-notification"]
                        ],
                        decls: 3,
                        vars: 0,
                        consts: [
                            [1, "container", "challenge-solved-toast", "mat-elevation-z4"],
                            ["appearance", "outlined", 1, "accent-notification"],
                            [1, "mdc-card"],
                            [1, "notificationMessage"],
                            ["id", "closeButton", "mat-button", "", 3, "click"],
                            [1, "icon-box"],
                            ["ngxClipboard", "", "mat-stroked-button", "", 3, "cbOnSuccess", "cbContent", "disabled"],
                            [3, "hidden"],
                            [3, "class"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "div", 0), t.Z7z(1, Ze, 7, 2, "mat-card", 1, We), t.k0s()), 2 & o && (t.R7$(), t.Dyx(i.notifications))
                        },
                        dependencies: [h.Hu, h.RN, b.Hl, b.$z, f.m_, f.An, Pt.FQ, Pt.Uo, d.h, Y.GH, d.D9],
                        styles: ["\n\n\n\n.container[_ngcontent-%COMP%]{font-size:14px;margin:40px}mat-card[_ngcontent-%COMP%]{margin-bottom:10px}mat-card[_ngcontent-%COMP%]   .notificationMessage[_ngcontent-%COMP%]{align-items:center;display:flex;justify-content:space-between}#closeButton[_ngcontent-%COMP%]{float:right}.icon-box[_ngcontent-%COMP%]{display:inline-flex;vertical-align:middle}"]
                    })
                }
                return n
            })();

            function en(n, a) {
                1 & n && (t.j41(0, "span", 1), t.EFF(1, "RESTART_REQUIRED"), t.k0s())
            }

            function nn(n, a) {
                1 & n && (t.j41(0, "span", 1), t.EFF(1, "RESET_HACKING_PROGRESS"), t.k0s())
            }

            function on(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-card", 0)(1, "mat-card-content")(2, "span", 1), t.EFF(3, "NOTIFICATION_SERVER_STARTED"), t.k0s(), t.EFF(4), t.j41(5, "button", 2), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.closeNotification())
                    }), t.EFF(6, "X"), t.k0s(), t.j41(7, "button", 3), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.clearProgress())
                    }), t.j41(8, "mat-icon"), t.EFF(9, "delete_forever"), t.k0s(), t.nVh(10, en, 2, 0, "span", 1)(11, nn, 2, 0, "span", 1), t.k0s()()()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.R7$(4), t.SpI(": ", e.hackingProgress.autoRestoreMessage, "\xa0"), t.R7$(3), t.Y8G("disabled", e.hackingProgress.cleared), t.R7$(3), t.vxM(e.hackingProgress.cleared ? 10 : 11)
                }
            }
            let an = (() => {
                class n {
                    ngZone;
                    challengeService;
                    translate;
                    cookieService;
                    ref;
                    io;
                    hackingProgress = {};
                    constructor(e, o, i, r, c, p) {
                        this.ngZone = e, this.challengeService = o, this.translate = i, this.cookieService = r, this.ref = c, this.io = p
                    }
                    ngOnInit() {
                        this.ngZone.runOutsideAngular(() => {
                            this.io.socket().on("server started", () => {
                                const e = this.cookieService.get("continueCode"),
                                    o = this.cookieService.get("continueCodeFindIt"),
                                    i = this.cookieService.get("continueCodeFixIt");
                                e && this.challengeService.restoreProgress(encodeURIComponent(e)).subscribe({
                                    next: () => {
                                        this.translate.get("AUTO_RESTORED_PROGRESS").subscribe({
                                            next: r => {
                                                this.hackingProgress.autoRestoreMessage = r
                                            },
                                            error: r => {
                                                this.hackingProgress.autoRestoreMessage = r
                                            }
                                        })
                                    },
                                    error: r => {
                                        console.log(r), this.translate.get("AUTO_RESTORE_PROGRESS_FAILED", {
                                            error: r
                                        }).subscribe({
                                            next: c => {
                                                this.hackingProgress.autoRestoreMessage = c
                                            },
                                            error: c => {
                                                this.hackingProgress.autoRestoreMessage = c
                                            }
                                        })
                                    }
                                }), o && this.challengeService.restoreProgressFindIt(encodeURIComponent(o)).subscribe({
                                    next: () => {},
                                    error: r => {
                                        console.log(r)
                                    }
                                }), i && this.challengeService.restoreProgressFixIt(encodeURIComponent(i)).subscribe({
                                    next: () => {},
                                    error: r => {
                                        console.log(r)
                                    }
                                }), this.ref.detectChanges()
                            })
                        })
                    }
                    closeNotification() {
                        this.hackingProgress.autoRestoreMessage = null
                    }
                    clearProgress() {
                        this.cookieService.remove("continueCode"), this.cookieService.remove("continueCodeFixIt"), this.cookieService.remove("continueCodeFindIt"), this.cookieService.remove("token"), sessionStorage.removeItem("bid"), sessionStorage.removeItem("itemTotal"), localStorage.removeItem("token"), localStorage.removeItem("displayedDifficulties"), localStorage.removeItem("showSolvedChallenges"), localStorage.removeItem("showDisabledChallenges"), localStorage.removeItem("showOnlyTutorialChallenges"), localStorage.removeItem("displayedChallengeCategories"), this.hackingProgress.cleared = !0
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(t.SKi), t.rXU(Z), t.rXU(d.c$), t.rXU($.O1), t.rXU(D.gRc), t.rXU(et))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-server-started-notification"]
                        ],
                        decls: 1,
                        vars: 1,
                        consts: [
                            ["appearance", "outlined", 1, "container", "primary-notification", "mat-elevation-z4"],
                            ["translate", ""],
                            ["id", "closeButton", "mat-button", "", 3, "click"],
                            ["mat-stroked-button", "", 3, "click", "disabled"]
                        ],
                        template: function(o, i) {
                            1 & o && t.nVh(0, on, 12, 3, "mat-card", 0), 2 & o && t.vxM(i.hackingProgress.autoRestoreMessage ? 0 : -1)
                        },
                        dependencies: [h.Hu, h.RN, h.m2, d.h, d.Mm, b.Hl, b.$z, f.m_, f.An],
                        styles: ["\n\n\n\n.container[_ngcontent-%COMP%]{font-size:14px;margin:40px}#closeButton[_ngcontent-%COMP%]{float:right}"]
                    })
                }
                return n
            })();
            var ne = m(1413);
            let P = (() => {
                    class n {
                        http;
                        isLoggedIn = new ne.B;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/api/Users";
                        constructor(e) {
                            this.http = e
                        }
                        find(e) {
                            return this.http.get(this.hostServer + "/rest/user/authentication-details/", {
                                params: e
                            }).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        get(e) {
                            return this.http.get(`${this.host}/${e}`).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        save(e) {
                            return this.http.post(this.host + "/", e).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        login(e) {
                            return this.isLoggedIn.next(!0), this.http.post(this.hostServer + "/rest/user/login", e).pipe((0, v.T)(o => o.authentication), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        getLoggedInState() {
                            return this.isLoggedIn.asObservable()
                        }
                        changePassword(e) {
                            return this.http.get(this.hostServer + "/rest/user/change-password?current=" + e.current + "&new=" + e.new + "&repeat=" + e.repeat).pipe((0, v.T)(o => o.user), (0, g.W)(o => {
                                throw o.error
                            }))
                        }
                        resetPassword(e) {
                            return this.http.post(this.hostServer + "/rest/user/reset-password", e).pipe((0, v.T)(o => o.user), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        whoAmI() {
                            return this.http.get(this.hostServer + "/rest/user/whoami").pipe((0, v.T)(e => e.user), (0, g.W)(e => {
                                throw e
                            }))
                        }
                        oauthLogin(e) {
                            return this.http.get("https://www.googleapis.com/oauth2/v1/userinfo?alt=json&access_token=" + e)
                        }
                        saveLastLoginIp() {
                            return this.http.get(this.hostServer + "/rest/saveLoginIp").pipe((0, v.T)(e => e), (0, g.W)(e => {
                                throw e
                            }))
                        }
                        deluxeStatus() {
                            return this.http.get(this.hostServer + "/rest/deluxe-membership").pipe((0, v.T)(e => e.data), (0, g.W)(e => {
                                throw e
                            }))
                        }
                        upgradeToDeluxe(e, o) {
                            return this.http.post(this.hostServer + "/rest/deluxe-membership", {
                                paymentMode: e,
                                paymentId: o
                            }).pipe((0, v.T)(i => i.data), (0, g.W)(i => {
                                throw i
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })(),
                jt = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/rest/admin";
                        constructor(e) {
                            this.http = e
                        }
                        getApplicationVersion() {
                            return this.http.get(this.host + "/application-version").pipe((0, v.T)(e => e.version), (0, g.W)(e => {
                                throw e
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })();
            var nt = m(8132);
            let rn = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        constructor(e) {
                            this.http = e
                        }
                        getLanguages() {
                            return this.http.get(`${this.hostServer}/rest/languages`).pipe((0, g.W)(e => {
                                throw e
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })(),
                Q = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        itemTotal = new ne.B;
                        host = this.hostServer + "/api/BasketItems";
                        constructor(e) {
                            this.http = e
                        }
                        find(e) {
                            return this.http.get(`${this.hostServer}/rest/basket/${e}`).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        get(e) {
                            return this.http.get(`${this.host}/${e}`).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        put(e, o) {
                            return this.http.put(`${this.host}/${e}`, o).pipe((0, v.T)(i => i.data), (0, g.W)(i => {
                                throw i
                            }))
                        }
                        del(e) {
                            return this.http.delete(`${this.host}/${e}`).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        save(e) {
                            return this.http.post(this.host + "/", e).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        checkout(e, o, i) {
                            return this.http.post(`${this.hostServer}/rest/basket/${e}/checkout`, {
                                couponData: o,
                                orderDetails: i
                            }).pipe((0, v.T)(r => r.orderConfirmation), (0, g.W)(r => {
                                throw r
                            }))
                        }
                        applyCoupon(e, o) {
                            return this.http.put(`${this.hostServer}/rest/basket/${e}/coupon/${o}`, {}).pipe((0, v.T)(i => i.discount), (0, g.W)(i => {
                                throw i
                            }))
                        }
                        updateNumberOfCartItems() {
                            this.find(parseInt(sessionStorage.getItem("bid"), 10)).subscribe({
                                next: e => {
                                    this.itemTotal.next(e.Products.reduce((o, i) => o + i.BasketItem.quantity, 0))
                                },
                                error: e => {
                                    console.log(e)
                                }
                            })
                        }
                        getItemTotal() {
                            return this.itemTotal.asObservable()
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })();
            var l = m(9417),
                I = m(1228),
                T = m(3746),
                C = m(9711),
                H = m(7810),
                N = m(6556),
                oe = m(9946);
            let X = (() => {
                    class n {
                        router;
                        ngZone;
                        constructor(e, o) {
                            this.router = e, this.ngZone = o
                        }
                        canActivate() {
                            return !!localStorage.getItem("token") || (this.forbidRoute("UNAUTHORIZED_ACCESS_ERROR"), !1)
                        }
                        forbidRoute(e = "UNAUTHORIZED_PAGE_ACCESS_ERROR") {
                            var o = this;
                            this.ngZone.run((0, F.A)(function*() {
                                return yield o.router.navigate(["403"], {
                                    skipLocationChange: !0,
                                    queryParams: {
                                        error: e
                                    }
                                })
                            }))
                        }
                        tokenDecode() {
                            let e = null;
                            const o = localStorage.getItem("token");
                            if (o) try {
                                e = oe(o)
                            } catch (i) {
                                console.log(i)
                            }
                            return e
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(R.Ix), s.KVO(t.SKi))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac
                        })
                    }
                    return n
                })(),
                ie = (() => {
                    class n {
                        loginGuard;
                        constructor(e) {
                            this.loginGuard = e
                        }
                        canActivate() {
                            const e = this.loginGuard.tokenDecode();
                            return !(!e?.data || "admin" !== e.data.role) || (this.loginGuard.forbidRoute(), !1)
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(X))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac
                        })
                    }
                    return n
                })(),
                ae = (() => {
                    class n {
                        loginGuard;
                        constructor(e) {
                            this.loginGuard = e
                        }
                        canActivate() {
                            const e = this.loginGuard.tokenDecode();
                            return !(!e?.data || "accounting" !== e.data.role) || (this.loginGuard.forbidRoute(), !1)
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(X))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac
                        })
                    }
                    return n
                })(),
                At = (() => {
                    class n {
                        loginGuard;
                        constructor(e) {
                            this.loginGuard = e
                        }
                        isDeluxe() {
                            const e = this.loginGuard.tokenDecode();
                            return e?.data && "deluxe" === e.data.role
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(X))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac
                        })
                    }
                    return n
                })();
            var U = m(1997),
                ot = m(5951),
                _t = m(9115),
                q = m(1514);
            class sn {
                value;
                onChange = a => {};
                onTouched = () => {};
                updateChanges() {
                    this.onChange(this.value)
                }
                writeValue(a) {
                    this.value = a, this.updateChanges()
                }
                registerOnChange(a) {
                    this.onChange = a
                }
                registerOnTouched(a) {
                    this.onTouched = a
                }
            }
            var ln = m(2496),
                _ = m(9588);
            const cn = ["input"];
            let dn = (() => {
                class n extends sn {
                    inputElement;
                    formControl;
                    matAutocomplete;
                    placeholder = "";
                    alwaysOpen = !1;
                    onBlur = new t.bkB;
                    onClose = new t.bkB;
                    onEnter = new t.bkB;
                    onFocus = new t.bkB;
                    onOpen = new t.bkB;
                    searchVisible = !1;
                    get isDisabled() {
                        return this.searchVisible ? null : "disabled"
                    }
                    ngOnInit() {
                        this.alwaysOpen && (this.searchVisible = !0)
                    }
                    close() {
                        this.alwaysOpen || (this.searchVisible = !1), this.value = "", this.updateChanges(), this.onClose.emit()
                    }
                    open() {
                        this.searchVisible = !0, this.inputElement.nativeElement.focus(), this.onOpen.emit()
                    }
                    onBlurring(e) {
                        !e && !this.alwaysOpen && (this.searchVisible = !1), this.onBlur.emit(e)
                    }
                    onEnterring(e) {
                        this.onEnter.emit(e)
                    }
                    onFocussing(e) {
                        this.onFocus.emit(e)
                    }
                    static \u0275fac = (() => {
                        let e;
                        return function(i) {
                            return (e || (e = t.xGo(n)))(i || n)
                        }
                    })();
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-mat-search-bar"]
                        ],
                        viewQuery: function(o, i) {
                            if (1 & o && t.GBs(cn, 5), 2 & o) {
                                let r;
                                t.mGM(r = t.lsd()) && (i.inputElement = r.first)
                            }
                        },
                        inputs: {
                            formControl: "formControl",
                            matAutocomplete: "matAutocomplete",
                            placeholder: "placeholder",
                            alwaysOpen: "alwaysOpen"
                        },
                        outputs: {
                            onBlur: "onBlur",
                            onClose: "onClose",
                            onEnter: "onEnter",
                            onFocus: "onFocus",
                            onOpen: "onOpen"
                        },
                        features: [t.Jv_([{
                            provide: l.kq,
                            useExisting: (0, s.Rfq)(() => n),
                            multi: !0
                        }]), t.Vt3],
                        decls: 8,
                        vars: 6,
                        consts: [
                            ["input", ""],
                            [1, "mat-search_field"],
                            ["matInput", "", "type", "text", "autocapitalize", "none", 3, "ngModelChange", "blur", "keyup.enter", "focus", "ngModel", "placeholder", "disabled"],
                            [1, "mat-search_icons"],
                            ["matRipple", "", 1, "mat-search_icon-close", 3, "click"],
                            ["matRipple", "", 1, "mat-search_icon-search", 3, "click"]
                        ],
                        template: function(o, i) {
                            if (1 & o) {
                                const r = t.RV6();
                                t.j41(0, "mat-form-field", 1)(1, "input", 2, 0), t.mxI("ngModelChange", function(p) {
                                    return s.eBV(r), t.DH7(i.value, p) || (i.value = p), s.Njj(p)
                                }), t.bIt("ngModelChange", function() {
                                    return s.eBV(r), s.Njj(i.updateChanges())
                                })("blur", function() {
                                    s.eBV(r);
                                    const p = t.sdS(2);
                                    return s.Njj(i.onBlurring(p.value))
                                })("keyup.enter", function() {
                                    s.eBV(r);
                                    const p = t.sdS(2);
                                    return s.Njj(i.onEnterring(p.value))
                                })("focus", function() {
                                    s.eBV(r);
                                    const p = t.sdS(2);
                                    return s.Njj(i.onFocussing(p.value))
                                }), t.k0s()(), t.j41(3, "span", 3)(4, "mat-icon", 4), t.bIt("click", function() {
                                    return s.eBV(r), s.Njj(i.close())
                                }), t.EFF(5, " close "), t.k0s(), t.j41(6, "mat-icon", 5), t.bIt("click", function() {
                                    return s.eBV(r), s.Njj(i.open())
                                }), t.EFF(7, " search "), t.k0s()()
                            }
                            2 & o && (t.Y8G("@slideInOut", i.searchVisible), t.R7$(), t.R50("ngModel", i.value), t.Y8G("placeholder", i.placeholder)("disabled", i.isDisabled), t.R7$(2), t.AVh("mat-search_icons--active", i.searchVisible))
                        },
                        dependencies: [I.R, _.rl, T.fS, T.fg, l.YN, l.me, l.BC, l.vS, f.m_, f.An, ln.r6],
                        styles: ["[_nghost-%COMP%]{align-items:center;display:flex;flex-direction:row;height:24px;justify-content:flex-end;position:relative;width:24px}.mat-search_field[_ngcontent-%COMP%]{padding:0 24px 0 0;position:absolute;z-index:1}.mat-search_field[_ngcontent-%COMP%]     .mat-mdc-text-field-wrapper{padding:0}.mat-search_field[_ngcontent-%COMP%]     .mat-mdc-form-field-infix{align-items:flex-end;display:flex;padding:.5em}.mat-search_icons[_ngcontent-%COMP%]{display:block;left:0;margin:0;padding:0;position:absolute;top:0}.mat-search_icons[_ngcontent-%COMP%]:hover{cursor:pointer}.mat-search_icons[_ngcontent-%COMP%]   .mat-search_icon-close[_ngcontent-%COMP%], .mat-search_icons[_ngcontent-%COMP%]   .mat-search_icon-search[_ngcontent-%COMP%]{border-radius:50%;left:0;position:absolute;top:0;transition:all .2s cubic-bezier(.4,0,.2,1);z-index:2}.mat-search_icons[_ngcontent-%COMP%]   .mat-search_icon-close[_ngcontent-%COMP%], .mat-search_icons.mat-search_icons--active[_ngcontent-%COMP%]   .mat-search_icon-search[_ngcontent-%COMP%]{opacity:0;transform:rotate(225deg)}.mat-search_icons.mat-search_icons--active[_ngcontent-%COMP%]   .mat-search_icon-close[_ngcontent-%COMP%]{opacity:1;transform:rotate(360deg);visibility:visible;z-index:3}"],
                        data: {
                            animation: [(0, q.hZ)("slideInOut", [(0, q.wk)("true", (0, q.iF)({
                                width: "*"
                            })), (0, q.wk)("false", (0, q.iF)({
                                width: "0"
                            })), (0, q.kY)("true => false", (0, q.i0)("300ms ease-in")), (0, q.kY)("false => true", (0, q.i0)("300ms ease-out"))])]
                        }
                    })
                }
                return n
            })();
            var dt = m(5911),
                rt = m(5416),
                V = m(2598);
            const mn = () => ["privacy-security/privacy-policy"],
                pn = () => ["privacy-security/data-export"],
                un = () => ["privacy-security/change-password"],
                hn = () => ["privacy-security/two-factor-authentication"],
                gn = () => ["privacy-security/last-login-ip"],
                fn = (n, a) => a.key;

            function _n(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 43), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.goToProfilePage())
                    }), t.j41(1, "mat-icon"), t.EFF(2, " account_circle "), t.k0s(), t.j41(3, "span"), t.EFF(4), t.k0s()()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.R7$(4), t.SpI(" ", e.userEmail, " ")
                }
            }

            function Cn(n, a) {
                1 & n && (t.j41(0, "button", 17)(1, "mat-icon"), t.EFF(2, " exit_to_app "), t.k0s(), t.j41(3, "span"), t.EFF(4), t.nI1(5, "translate"), t.k0s()()), 2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "TITLE_LOGIN"), " "))
            }

            function bn(n, a) {
                1 & n && (t.j41(0, "button", 18)(1, "mat-icon"), t.EFF(2, " account_balance "), t.k0s(), t.j41(3, "span"), t.EFF(4), t.nI1(5, "translate"), t.k0s()()), 2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "ACCOUNTING"), " "))
            }

            function vn(n, a) {
                if (1 & n && (t.j41(0, "button", 19)(1, "mat-icon"), t.EFF(2, " check_circle_outline "), t.k0s(), t.j41(3, "span"), t.EFF(4), t.nI1(5, "translate"), t.k0s()()), 2 & n) {
                    t.XpG();
                    const e = t.sdS(83);
                    t.Y8G("matMenuTriggerFor", e), t.R7$(4), t.SpI(" ", t.bMT(5, 2, "ORDERS_AND_PAYMENT"), " ")
                }
            }

            function kn(n, a) {
                if (1 & n && (t.j41(0, "button", 20)(1, "mat-icon"), t.EFF(2, " security "), t.k0s(), t.j41(3, "span"), t.EFF(4), t.nI1(5, "translate"), t.k0s()()), 2 & n) {
                    t.XpG();
                    const e = t.sdS(50);
                    t.Y8G("matMenuTriggerFor", e), t.R7$(4), t.SpI(" ", t.bMT(5, 2, "PRIVACY_AND_SECURITY"), " ")
                }
            }

            function Sn(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 44), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.logout())
                    }), t.j41(1, "mat-icon"), t.EFF(2, " power_settings_new "), t.k0s(), t.j41(3, "span"), t.EFF(4), t.nI1(5, "translate"), t.k0s()()
                }
                2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "TITLE_LOGOUT"), " "))
            }

            function Fn(n, a) {
                if (1 & n && (t.j41(0, "button", 22)(1, "mat-icon"), t.EFF(2, " shopping_cart "), t.k0s(), t.j41(3, "span", 45), t.EFF(4), t.nI1(5, "translate"), t.k0s(), t.j41(6, "span", 46), t.EFF(7), t.k0s()()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(4), t.SpI(" ", t.bMT(5, 2, "TITLE_BASKET")), t.R7$(3), t.JRh(e.itemTotal)
                }
            }

            function xn(n, a) {
                1 & n && t.nrm(0, "span"), 2 & n && t.HbH("fi fi-" + a.$implicit)
            }

            function Tn(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-radio-button", 47), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG();
                        return s.Njj(r.changeLanguage(i.key))
                    }), t.j41(1, "div", 48), t.Z7z(2, xn, 1, 2, "span", 49, t.fX1), t.EFF(4), t.k0s(), t.nrm(5, "i"), t.k0s()
                }
                if (2 & n) {
                    const e = a.$implicit,
                        o = t.XpG();
                    t.Y8G("value", e)("checked", o.selectedLanguage === e), t.jOp("aria-label", e.lang), t.R7$(2), t.Dyx(e.icons), t.R7$(2), t.SpI(" ", null == e ? null : e.lang, " "), t.R7$(), t.HbH("fas fa-thermometer-" + e.gauge + (e.percentage > 70 ? " confirmation" : " error"))
                }
            }

            function yn(n, a) {
                1 & n && (t.j41(0, "button", 40)(1, "mat-icon"), t.EFF(2, " my_location "), t.k0s(), t.j41(3, "span"), t.EFF(4), t.nI1(5, "translate"), t.k0s()()), 2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "MY_SAVED_ADRESSES"), " "))
            }

            function Rn(n, a) {
                1 & n && (t.j41(0, "button", 41)(1, "mat-icon"), t.EFF(2, " credit_card "), t.k0s(), t.j41(3, "span"), t.EFF(4), t.nI1(5, "translate"), t.k0s()()), 2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "MY_PAYMENT_OPTIONS"), " "))
            }

            function wn(n, a) {
                1 & n && (t.j41(0, "button", 42)(1, "mat-icon"), t.EFF(2, " account_balance_wallet "), t.k0s(), t.j41(3, "span"), t.EFF(4), t.nI1(5, "translate"), t.k0s()()), 2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "DIGITAL_WALLET"), " "))
            }
            S.Yv.add(C.mh, C.MjD, C.rhE, C.GxD, C.DN2, C.iyL, C.iHh, C.iW_, C._Ro, C.LkM, C.v3E, C.sFQ, C.VFr, N.Vz1, H.q9, C.zfV, C.sfq, C.Rog, C.H77, C.rlR);
            let En = (() => {
                class n {
                    administrationService;
                    challengeService;
                    configurationService;
                    userService;
                    ngZone;
                    cookieService;
                    router;
                    translate;
                    io;
                    langService;
                    loginGuard;
                    snackBar;
                    basketService;
                    userEmail = "";
                    languages = [];
                    filteredLanguages = [];
                    languageSearchQuery = "";
                    selectedLanguage = "placeholder";
                    version = "";
                    applicationName = "OWASP Juice Shop";
                    showGitHubLink = !0;
                    logoSrc = "assets/public/images/JuiceShop_Logo.png";
                    scoreBoardVisible = !1;
                    shortKeyLang = "placeholder";
                    itemTotal = 0;
                    sidenavToggle = new t.bkB;
                    constructor(e, o, i, r, c, p, k, x, O, K, ht, gt, ft) {
                        this.administrationService = e, this.challengeService = o, this.configurationService = i, this.userService = r, this.ngZone = c, this.cookieService = p, this.router = k, this.translate = x, this.io = O, this.langService = K, this.loginGuard = ht, this.snackBar = gt, this.basketService = ft
                    }
                    ngOnInit() {
                        this.getLanguages(), this.basketService.getItemTotal().subscribe(e => this.itemTotal = e), this.administrationService.getApplicationVersion().subscribe({
                            next: e => {
                                e && (this.version = `v${e}`)
                            },
                            error: e => {
                                console.log(e)
                            }
                        }), this.configurationService.getApplicationConfiguration().subscribe({
                            next: e => {
                                if (e?.application?.name && (this.applicationName = e.application.name), e?.application && (this.showGitHubLink = e.application.showGitHubLinks), e?.application?.logo) {
                                    let o = e.application.logo;
                                    "http" === o.substring(0, 4) && (o = decodeURIComponent(o.substring(o.lastIndexOf("/") + 1))), this.logoSrc = "assets/public/images/" + o
                                }
                            },
                            error: e => {
                                console.log(e)
                            }
                        }), localStorage.getItem("token") ? this.getUserDetails() : this.userEmail = "", this.userService.getLoggedInState().subscribe(e => {
                            e ? this.getUserDetails() : this.userEmail = ""
                        }), this.getScoreBoardStatus(), this.ngZone.runOutsideAngular(() => {
                            this.io.socket().on("challenge solved", e => {
                                "scoreBoardChallenge" === e.key && (this.scoreBoardVisible = !0)
                            })
                        })
                    }
                    filterLanguages() {
                        if (!this.languageSearchQuery) return void(this.filteredLanguages = [...this.languages]);
                        const e = this.languageSearchQuery.toLowerCase();
                        this.filteredLanguages = this.languages.filter(o => !!(o.lang.toLowerCase().includes(e) || o.key.toLowerCase().includes(e) || o.shortKey?.toLowerCase()?.includes(e)))
                    }
                    checkLanguage() {
                        if (this.cookieService.get("language")) {
                            const e = this.cookieService.get("language");
                            this.translate.use(e), this.selectedLanguage = this.languages.find(o => o.key === e), this.shortKeyLang = this.languages.find(o => o.key === e).shortKey
                        } else this.changeLanguage("en"), this.selectedLanguage = this.languages.find(e => "en" === e.key), this.shortKeyLang = this.languages.find(e => "en" === e.key).shortKey
                    }
                    search(e) {
                        var o = this;
                        if (e) {
                            const i = {
                                queryParams: {
                                    q: e
                                }
                            };
                            this.ngZone.run((0, F.A)(function*() {
                                return yield o.router.navigate(["/search"], i)
                            }))
                        } else this.ngZone.run((0, F.A)(function*() {
                            return yield o.router.navigate(["/search"])
                        }))
                    }
                    getUserDetails() {
                        this.userService.whoAmI().subscribe({
                            next: e => {
                                this.userEmail = e.email
                            },
                            error: e => {
                                console.log(e)
                            }
                        })
                    }
                    isLoggedIn() {
                        return localStorage.getItem("token")
                    }
                    logout() {
                        var e = this;
                        this.userService.saveLastLoginIp().subscribe({
                            next: o => {
                                this.noop()
                            },
                            error: o => {
                                console.log(o)
                            }
                        }), localStorage.removeItem("token"), this.cookieService.remove("token"), sessionStorage.removeItem("bid"), sessionStorage.removeItem("itemTotal"), this.userService.isLoggedIn.next(!1), this.ngZone.run((0, F.A)(function*() {
                            return yield e.router.navigate(["/"])
                        }))
                    }
                    changeLanguage(e) {
                        this.translate.use(e);
                        const o = new Date;
                        if (o.setFullYear(o.getFullYear() + 1), this.cookieService.put("language", e, {
                                expires: o
                            }), this.languages.find(i => i.key === e)) {
                            const i = this.languages.find(c => c.key === e);
                            this.shortKeyLang = i.shortKey, this.snackBar.open(`Language has been changed to ${i.lang}`, "Force page reload", {
                                duration: 5e3,
                                panelClass: ["mat-body"]
                            }).onAction().subscribe(() => {
                                location.reload()
                            })
                        }
                    }
                    getScoreBoardStatus() {
                        this.challengeService.find({
                            name: "Score Board"
                        }).subscribe({
                            next: e => {
                                this.ngZone.run(() => {
                                    this.scoreBoardVisible = e[0].solved
                                })
                            },
                            error: e => {
                                console.log(e)
                            }
                        })
                    }
                    goToProfilePage() {
                        window.location.replace(y.c.hostServer + "/profile")
                    }
                    goToDataErasurePage() {
                        window.location.replace(y.c.hostServer + "/dataerasure")
                    }
                    onToggleSidenav = () => {
                        this.sidenavToggle.emit()
                    };
                    noop() {}
                    getLanguages() {
                        this.langService.getLanguages().subscribe(e => {
                            this.languages = e, this.filteredLanguages = Array.isArray(e) ? [...e] : [], this.checkLanguage()
                        })
                    }
                    isAccounting() {
                        const e = this.loginGuard.tokenDecode();
                        return e?.data && "accounting" === e.data.role
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(jt), t.rXU(Z), t.rXU(M.s), t.rXU(P), t.rXU(t.SKi), t.rXU($.O1), t.rXU(R.Ix), t.rXU(d.c$), t.rXU(et), t.rXU(rn), t.rXU(X), t.rXU(rt.UG), t.rXU(Q))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-navbar"]
                        ],
                        outputs: {
                            sidenavToggle: "sidenavToggle"
                        },
                        decls: 100,
                        vars: 48,
                        consts: [
                            ["searchControl", ""],
                            ["userMenu", "matMenu"],
                            ["menu", "matMenu"],
                            ["privacySubMenu", "matMenu"],
                            ["ordersSubMenu", "matMenu"],
                            ["color", "primary", 1, "mat-elevation-z6", "navbar-toolbar"],
                            [1, "navbar-row"],
                            ["mat-icon-button", "", "aria-label", "Open Sidenav", "matTooltipPosition", "below", 2, "height", "48px", "width", "48px", 3, "click", "matTooltip"],
                            ["mat-button", "", "routerLink", "/search", "aria-label", "Back to homepage", 1, "buttons", 2, "height", "60px"],
                            ["id", "homeButton"],
                            [1, "logo", 3, "src", "alt"],
                            [1, "hide-lt-sm", 2, "font-size", "x-large"],
                            [1, "fill-remaining-space"],
                            ["id", "product-search-fixture"],
                            ["id", "searchQuery", "aria-label", "Click to search", 3, "onEnter"],
                            ["mat-button", "", "aria-label", "Show/hide account menu", "id", "navbarAccount", 1, "buttons", "hide-lt-md", 2, "vertical-align", "middle", "height", "48px", 3, "matMenuTriggerFor"],
                            ["mat-menu-item", "", "aria-label", "Go to user profile"],
                            ["mat-menu-item", "", "routerLink", "/login", "aria-label", "Go to login page", "id", "navbarLoginButton"],
                            ["mat-menu-item", "", "routerLink", "/accounting", "aria-label", "Go to accounting page"],
                            ["mat-menu-item", "", "aria-label", "Show Orders and Payment Menu", 3, "matMenuTriggerFor"],
                            ["mat-menu-item", "", "aria-label", "Show Privacy and Security Menu", 3, "matMenuTriggerFor"],
                            ["mat-menu-item", "", "aria-label", "Logout", "id", "navbarLogoutButton"],
                            ["mat-button", "", "routerLink", "/basket", "aria-label", "Show the shopping cart", 1, "buttons", 2, "height", "48px"],
                            ["id", "navbarLanguageButton", "mat-button", "", "aria-label", "Language selection menu", "matTooltipPosition", "below", 1, "buttons", 2, "height", "48px", "width", "48px", 3, "matMenuTriggerFor", "matTooltip"],
                            [1, "hide-lt-md"],
                            [3, "overlapTrigger"],
                            [1, "language-search-container", 3, "click"],
                            [1, "language-search-field"],
                            ["matPrefix", "", 1, "search-icon"],
                            ["type", "search", "matInput", "", 3, "ngModelChange", "input", "ngModel"],
                            [1, "mat-menu-item", 2, "width", "240px", 3, "value", "checked", "aria-label"],
                            ["mat-menu-item", "", "aria-label", "Go to privacy policy page", 3, "routerLink"],
                            ["translate", ""],
                            ["mat-menu-item", "", "aria-label", "Go to data export page", 3, "routerLink"],
                            ["mat-menu-item", "", "aria-label", "Go to data subject page", 3, "click"],
                            ["mat-menu-item", "", "aria-label", "Go to change password page", 3, "routerLink"],
                            ["mat-menu-item", "", "aria-label", "Go to two factor authentication page", 3, "routerLink"],
                            ["mat-menu-item", "", "aria-label", "Go to last login ip page", 3, "routerLink"],
                            ["mat-menu-item", "", "routerLink", "/order-history", "aria-label", "Go to order history page"],
                            ["mat-menu-item", "", "routerLink", "/recycle", "aria-label", "Go to recycling page"],
                            ["mat-menu-item", "", "routerLink", "/address/saved", "aria-label", "Go to saved address page"],
                            ["mat-menu-item", "", "routerLink", "/saved-payment-methods", "aria-label", "Go to saved payment methods page"],
                            ["mat-menu-item", "", "routerLink", "/wallet", "aria-label", "Go to wallet page"],
                            ["mat-menu-item", "", "aria-label", "Go to user profile", 3, "click"],
                            ["mat-menu-item", "", "aria-label", "Logout", "id", "navbarLogoutButton", 3, "click"],
                            [1, "hide-lt-md", 2, "margin-right", "10px"],
                            [1, "fa-layers-counter", "fa-layers-top-right", "fa-3x", "warn-notification", 2, "font-size", "47px"],
                            [1, "mat-menu-item", 2, "width", "240px", 3, "click", "value", "checked", "aria-label"],
                            [1, "mat-body", 2, "display", "inline-block", "width", "200px", "margin-left", "5px"],
                            [3, "class"]
                        ],
                        template: function(o, i) {
                            if (1 & o) {
                                const r = t.RV6();
                                t.j41(0, "mat-toolbar", 5)(1, "mat-toolbar-row", 6)(2, "button", 7), t.nI1(3, "translate"), t.bIt("click", function() {
                                    return s.eBV(r), s.Njj(i.onToggleSidenav())
                                }), t.j41(4, "mat-icon"), t.EFF(5, "menu"), t.k0s()(), t.j41(6, "button", 8)(7, "div", 9), t.nrm(8, "img", 10), t.j41(9, "span", 11), t.EFF(10), t.k0s()()(), t.nrm(11, "span", 12), t.j41(12, "div"), t.nrm(13, "div", 13), t.k0s(), t.j41(14, "app-mat-search-bar", 14, 0), t.bIt("onEnter", function() {
                                    s.eBV(r);
                                    const p = t.sdS(15);
                                    return s.Njj(i.search(p.value))
                                }), t.k0s(), t.j41(16, "button", 15)(17, "mat-icon"), t.EFF(18, " account_circle "), t.k0s(), t.j41(19, "span"), t.EFF(20), t.nI1(21, "translate"), t.k0s()(), t.j41(22, "mat-menu", null, 1), t.nVh(24, _n, 5, 1, "button", 16)(25, Cn, 6, 3, "button", 17), t.nVh(26, bn, 6, 3, "button", 18), t.nVh(27, vn, 6, 4, "button", 19), t.nVh(28, kn, 6, 4, "button", 20), t.nVh(29, Sn, 6, 3, "button", 21), t.k0s(), t.nVh(30, Fn, 8, 4, "button", 22), t.j41(31, "button", 23), t.nI1(32, "translate"), t.j41(33, "mat-icon"), t.EFF(34, " language "), t.k0s(), t.j41(35, "span", 24), t.EFF(36), t.k0s()(), t.j41(37, "mat-menu", 25, 2)(39, "div", 26), t.bIt("click", function(p) {
                                    return s.eBV(r), s.Njj(p.stopPropagation())
                                }), t.j41(40, "mat-form-field", 27)(41, "mat-icon", 28), t.EFF(42, "search"), t.k0s(), t.j41(43, "mat-label"), t.EFF(44), t.nI1(45, "translate"), t.k0s(), t.j41(46, "input", 29), t.mxI("ngModelChange", function(p) {
                                    return s.eBV(r), t.DH7(i.languageSearchQuery, p) || (i.languageSearchQuery = p), s.Njj(p)
                                }), t.bIt("input", function() {
                                    return s.eBV(r), s.Njj(i.filterLanguages())
                                }), t.k0s()()(), t.Z7z(47, Tn, 6, 6, "mat-radio-button", 30, fn), t.k0s(), t.j41(49, "mat-menu", null, 3)(51, "button", 31)(52, "mat-icon"), t.EFF(53, " assignment "), t.k0s(), t.j41(54, "span", 32), t.EFF(55, "TITLE_PRIVACY_POLICY"), t.k0s()(), t.j41(56, "button", 33)(57, "mat-icon"), t.EFF(58, " get_app "), t.k0s(), t.j41(59, "span", 32), t.EFF(60, "TITLE_REQUEST_DATA_EXPORT"), t.k0s()(), t.j41(61, "button", 34), t.bIt("click", function() {
                                    return s.eBV(r), s.Njj(i.goToDataErasurePage())
                                }), t.j41(62, "mat-icon"), t.EFF(63, " delete_forever "), t.k0s(), t.j41(64, "span", 32), t.EFF(65, "DATA_SUBJECT_TITLE"), t.k0s()(), t.nrm(66, "mat-divider"), t.j41(67, "button", 35)(68, "mat-icon"), t.EFF(69, " edit "), t.k0s(), t.j41(70, "span", 32), t.EFF(71, "TITLE_CHANGE_PASSWORD"), t.k0s()(), t.j41(72, "button", 36)(73, "mat-icon"), t.EFF(74, " exposure_plus_2 "), t.k0s(), t.j41(75, "span", 32), t.EFF(76, "TITLE_TWO_FACTOR_AUTH_CONFIG"), t.k0s()(), t.j41(77, "button", 37)(78, "mat-icon"), t.EFF(79, " place "), t.k0s(), t.j41(80, "span", 32), t.EFF(81, "LAST_LOGIN_IP"), t.k0s()()(), t.j41(82, "mat-menu", null, 4)(84, "button", 38)(85, "mat-icon"), t.EFF(86, " archive "), t.k0s(), t.j41(87, "span"), t.EFF(88), t.nI1(89, "translate"), t.k0s()(), t.j41(90, "button", 39)(91, "mat-icon"), t.EFF(92, " autorenew "), t.k0s(), t.j41(93, "span"), t.EFF(94), t.nI1(95, "translate"), t.k0s()(), t.nrm(96, "mat-divider"), t.nVh(97, yn, 6, 3, "button", 40), t.nVh(98, Rn, 6, 3, "button", 41), t.nVh(99, wn, 6, 3, "button", 42), t.k0s()()()
                            }
                            if (2 & o) {
                                const r = t.sdS(23),
                                    c = t.sdS(38);
                                t.R7$(2), t.Y8G("matTooltip", t.mNQ(t.bMT(3, 31, "SIDENAV_HINT"))), t.R7$(6), t.Y8G("alt", t.mNQ(i.applicationName))("src", i.logoSrc, t.B4B), t.R7$(2), t.SpI(" ", i.applicationName, " "), t.R7$(6), t.Y8G("matMenuTriggerFor", r), t.R7$(4), t.SpI(" ", t.bMT(21, 33, "ACCOUNT"), " "), t.R7$(4), t.vxM(i.isLoggedIn() ? 24 : 25), t.R7$(2), t.vxM(i.isLoggedIn() && i.isAccounting() ? 26 : -1), t.R7$(), t.vxM(i.isLoggedIn() ? 27 : -1), t.R7$(), t.vxM(i.isLoggedIn() ? 28 : -1), t.R7$(), t.vxM(i.isLoggedIn() ? 29 : -1), t.R7$(), t.vxM(i.isLoggedIn() ? 30 : -1), t.R7$(), t.Y8G("matTooltip", t.mNQ(t.bMT(32, 35, "LANGUAGE_SEL_HINT")))("matMenuTriggerFor", c), t.R7$(5), t.SpI(" ", i.shortKeyLang, " "), t.R7$(), t.Y8G("overlapTrigger", !0), t.R7$(7), t.JRh(t.bMT(45, 37, "SEARCH")), t.R7$(2), t.R50("ngModel", i.languageSearchQuery), t.R7$(), t.Dyx(i.filteredLanguages), t.R7$(4), t.Y8G("routerLink", t.lJ4(43, mn)), t.R7$(5), t.Y8G("routerLink", t.lJ4(44, pn)), t.R7$(11), t.Y8G("routerLink", t.lJ4(45, un)), t.R7$(5), t.Y8G("routerLink", t.lJ4(46, hn)), t.R7$(5), t.Y8G("routerLink", t.lJ4(47, gn)), t.R7$(11), t.SpI(" ", t.bMT(89, 39, "LABEL_ORDER_HISTORY"), " "), t.R7$(6), t.SpI(" ", t.bMT(95, 41, "NAV_RECYCLE"), " "), t.R7$(3), t.vxM(i.isLoggedIn() ? 97 : -1), t.R7$(), t.vxM(i.isLoggedIn() ? 98 : -1), t.R7$(), t.vxM(i.isLoggedIn() ? 99 : -1)
                            }
                        },
                        dependencies: [dt.KQ, dt.uu, b.Hl, b.$z, V.iY, B.oV, f.m_, f.An, nt.Wk, dn, _t.Cp, _t.kk, _t.fb, ot._g, d.h, d.Mm, U.q, I.R, _.rl, _.nJ, _.JW, T.fS, T.fg, l.YN, l.me, l.BC, l.vS, d.D9],
                        styles: ["\n\n\n\nmat-toolbar[_ngcontent-%COMP%]{height:auto;min-width:100%;padding:4px 0;width:100%}.mat-mdc-button[_ngcontent-%COMP%]{padding-left:4px;padding-right:4px}.navbar-toolbar[_ngcontent-%COMP%]{display:flex;flex-direction:column}.navbar-row[_ngcontent-%COMP%]{align-items:center;display:flex}@media (max-width: 599.98px){.hide-lt-sm[_ngcontent-%COMP%]{display:none!important}}@media (max-width: 959.98px){.hide-lt-md[_ngcontent-%COMP%]{display:none!important}}#navbarLanguageButton[_ngcontent-%COMP%]   mat-icon[_ngcontent-%COMP%]{font-size:24px;height:26px;margin-right:2px;width:26px}mat-toolbar[_ngcontent-%COMP%]   mat-icon[_ngcontent-%COMP%]{font-size:23px;height:24px;width:24px}.warn-notification[_ngcontent-%COMP%]{left:15px;top:-15px}.logo[_ngcontent-%COMP%]{max-height:60px;width:auto}#homeButton[_ngcontent-%COMP%]{align-items:center;display:flex;min-width:0}.avatar[_ngcontent-%COMP%]{background-repeat:no-repeat no-repeat;border-radius:50%;display:inline-block;max-height:35px;max-width:35px;padding:0;pointer-events:none}.fill-remaining-space[_ngcontent-%COMP%]{flex:1 1 auto}.language-select[_ngcontent-%COMP%]{margin-left:10px;width:11%}.user-info[_ngcontent-%COMP%]{margin-left:10px;margin-right:10px}.fi[_ngcontent-%COMP%]{margin-right:2px}[_nghost-%COMP%]     mat-form-field{background:transparent!important}[_nghost-%COMP%]     mat-icon{background:transparent!important}mat-search-bar[_ngcontent-%COMP%]{font-size:13px;margin-right:80px;min-width:200px;padding-right:100px}.buttons[_ngcontent-%COMP%]{position:relative;width:auto}.language-search-container[_ngcontent-%COMP%]{padding:8px 16px}.language-search-field[_ngcontent-%COMP%]{width:100%}.search-icon[_ngcontent-%COMP%]{color:var(--theme-text-fade-30)}@media screen and (max-width: 959px){.buttons[_ngcontent-%COMP%]{width:48px}.warn-notification[_ngcontent-%COMP%]{left:-60px;position:absolute}}@media screen and (max-width: 959px) and (min-width: 600px){#homeButton[_ngcontent-%COMP%]{margin-left:8px}#homeButton[_ngcontent-%COMP%]   span[_ngcontent-%COMP%]{font-size:1.2rem!important;white-space:nowrap}button[mat-button][_ngcontent-%COMP%]{padding:0 8px}.buttons[_ngcontent-%COMP%]{min-width:48px;width:auto!important}}"]
                    })
                }
                return n
            })();
            var Ct = m(3902);
            const re = n => ({
                    rotated: n
                }),
                se = n => ({
                    expanded: n
                });

            function Mn(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 35), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.onToggleSidenav())
                    }), t.j41(1, "mat-icon"), t.EFF(2, " exit_to_app "), t.k0s(), t.j41(3, "span", 13), t.EFF(4), t.nI1(5, "translate"), t.k0s()()
                }
                2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "TITLE_LOGIN"), " "))
            }

            function In(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 36), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return i.onToggleSidenav(), s.Njj(i.goToProfilePage())
                    }), t.j41(1, "mat-icon"), t.EFF(2, " account_circle "), t.k0s(), t.j41(3, "span", 13), t.EFF(4), t.k0s()()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.R7$(4), t.SpI(" ", e.userEmail, " ")
                }
            }

            function On(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 37), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.onToggleSidenav())
                    }), t.j41(1, "mat-icon"), t.EFF(2, " account_balance "), t.k0s(), t.j41(3, "span", 13), t.EFF(4), t.nI1(5, "translate"), t.k0s()()
                }
                2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "ACCOUNTING"), " "))
            }

            function Pn(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-list-item", 38), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.showOrdersSubmenu = !i.showOrdersSubmenu)
                    }), t.j41(1, "mat-icon"), t.EFF(2, " check_circle_outline "), t.k0s(), t.j41(3, "span", 13), t.EFF(4), t.nI1(5, "translate"), t.k0s(), t.j41(6, "mat-icon", 39), t.EFF(7, " expand_more "), t.k0s()()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.R7$(4), t.SpI(" ", t.bMT(5, 2, "ORDERS_AND_PAYMENT"), " "), t.R7$(2), t.Y8G("ngClass", t.eq3(4, re, e.showOrdersSubmenu))
                }
            }

            function jn(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 45), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG(2);
                        return i.onToggleSidenav(), s.Njj(i.showOrdersSubmenu = !i.showOrdersSubmenu)
                    }), t.j41(1, "mat-icon"), t.EFF(2, " archive "), t.k0s(), t.j41(3, "span", 13), t.EFF(4), t.nI1(5, "translate"), t.k0s()()
                }
                2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "LABEL_ORDER_HISTORY"), " "))
            }

            function An(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 46), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG(2);
                        return i.onToggleSidenav(), s.Njj(i.showOrdersSubmenu = !i.showOrdersSubmenu)
                    }), t.j41(1, "mat-icon"), t.EFF(2, " autorenew "), t.k0s(), t.j41(3, "span", 13), t.EFF(4), t.nI1(5, "translate"), t.k0s()()
                }
                2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "NAV_RECYCLE"), " "))
            }

            function Dn(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 47), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG(2);
                        return i.onToggleSidenav(), s.Njj(i.showOrdersSubmenu = !i.showOrdersSubmenu)
                    }), t.j41(1, "mat-icon"), t.EFF(2, " my_location "), t.k0s(), t.j41(3, "span", 13), t.EFF(4), t.nI1(5, "translate"), t.k0s()()
                }
                2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "MY_SAVED_ADRESSES"), " "))
            }

            function $n(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 48), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG(2);
                        return i.onToggleSidenav(), s.Njj(i.showOrdersSubmenu = !i.showOrdersSubmenu)
                    }), t.j41(1, "mat-icon"), t.EFF(2, " credit_card "), t.k0s(), t.j41(3, "span", 13), t.EFF(4), t.nI1(5, "translate"), t.k0s()()
                }
                2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "MY_PAYMENT_OPTIONS"), " "))
            }

            function Ln(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 49), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG(2);
                        return i.onToggleSidenav(), s.Njj(i.showOrdersSubmenu = !i.showOrdersSubmenu)
                    }), t.j41(1, "mat-icon"), t.EFF(2, " account_balance_wallet "), t.k0s(), t.j41(3, "span", 13), t.EFF(4), t.nI1(5, "translate"), t.k0s()()
                }
                2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "DIGITAL_WALLET"), " "))
            }

            function Bn(n, a) {
                if (1 & n && (t.j41(0, "div", 8), t.nVh(1, jn, 6, 3, "a", 40), t.nVh(2, An, 6, 3, "a", 41), t.nrm(3, "mat-divider"), t.nVh(4, Dn, 6, 3, "a", 42), t.nVh(5, $n, 6, 3, "a", 43), t.nVh(6, Ln, 6, 3, "a", 44), t.k0s()), 2 & n) {
                    const e = t.XpG();
                    t.Y8G("ngClass", t.eq3(6, se, e.showOrdersSubmenu)), t.R7$(), t.vxM(e.isLoggedIn() ? 1 : -1), t.R7$(), t.vxM(e.isLoggedIn() ? 2 : -1), t.R7$(2), t.vxM(e.isLoggedIn() ? 4 : -1), t.R7$(), t.vxM(e.isLoggedIn() ? 5 : -1), t.R7$(), t.vxM(e.isLoggedIn() ? 6 : -1)
                }
            }

            function Nn(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-list-item", 50), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.showPrivacySubmenu = !i.showPrivacySubmenu)
                    }), t.j41(1, "mat-icon"), t.EFF(2, " security "), t.k0s(), t.j41(3, "span", 13), t.EFF(4), t.nI1(5, "translate"), t.k0s(), t.j41(6, "mat-icon", 39), t.EFF(7, " expand_more "), t.k0s()()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.R7$(4), t.SpI(" ", t.bMT(5, 2, "PRIVACY_AND_SECURITY"), " "), t.R7$(2), t.Y8G("ngClass", t.eq3(4, re, e.showPrivacySubmenu))
                }
            }

            function Un(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 57), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG(2);
                        return i.onToggleSidenav(), s.Njj(i.showPrivacySubmenu = !i.showPrivacySubmenu)
                    }), t.j41(1, "mat-icon"), t.EFF(2, " assignment "), t.k0s(), t.j41(3, "span", 58), t.EFF(4, "TITLE_PRIVACY_POLICY"), t.k0s()()
                }
            }

            function Vn(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 59), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG(2);
                        return i.onToggleSidenav(), s.Njj(i.showPrivacySubmenu = !i.showPrivacySubmenu)
                    }), t.j41(1, "mat-icon"), t.EFF(2, " get_app "), t.k0s(), t.j41(3, "span", 58), t.EFF(4, "TITLE_REQUEST_DATA_EXPORT"), t.k0s()()
                }
            }

            function Gn(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 60), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG(2);
                        return s.Njj(i.goToDataErasurePage())
                    })("click", function() {
                        s.eBV(e);
                        const i = t.XpG(2);
                        return i.onToggleSidenav(), s.Njj(i.showPrivacySubmenu = !i.showPrivacySubmenu)
                    }), t.j41(1, "mat-icon"), t.EFF(2, " delete_forever "), t.k0s(), t.j41(3, "span", 58), t.EFF(4, "DATA_SUBJECT_TITLE"), t.k0s()()
                }
            }

            function Yn(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 61), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG(2);
                        return i.onToggleSidenav(), s.Njj(i.showPrivacySubmenu = !i.showPrivacySubmenu)
                    }), t.j41(1, "mat-icon"), t.EFF(2, " edit "), t.k0s(), t.j41(3, "span", 58), t.EFF(4, "TITLE_CHANGE_PASSWORD"), t.k0s()()
                }
            }

            function Hn(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 62), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG(2);
                        return i.onToggleSidenav(), s.Njj(i.showPrivacySubmenu = !i.showPrivacySubmenu)
                    }), t.j41(1, "mat-icon"), t.EFF(2, " exposure_plus_2 "), t.k0s(), t.j41(3, "span", 58), t.EFF(4, "TITLE_TWO_FACTOR_AUTH_CONFIG"), t.k0s()()
                }
            }

            function Xn(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 63), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG(2);
                        return i.onToggleSidenav(), s.Njj(i.showPrivacySubmenu = !i.showPrivacySubmenu)
                    }), t.j41(1, "mat-icon"), t.EFF(2, " place "), t.k0s(), t.j41(3, "span", 58), t.EFF(4, "LAST_LOGIN_IP"), t.k0s()()
                }
            }

            function Wn(n, a) {
                if (1 & n && (t.j41(0, "div", 8), t.nVh(1, Un, 5, 0, "a", 51), t.nVh(2, Vn, 5, 0, "a", 52), t.nVh(3, Gn, 5, 0, "a", 53), t.nrm(4, "mat-divider"), t.nVh(5, Yn, 5, 0, "a", 54), t.nVh(6, Hn, 5, 0, "a", 55), t.nVh(7, Xn, 5, 0, "a", 56), t.k0s()), 2 & n) {
                    const e = t.XpG();
                    t.Y8G("ngClass", t.eq3(7, se, e.showPrivacySubmenu)), t.R7$(), t.vxM(e.isLoggedIn() ? 1 : -1), t.R7$(), t.vxM(e.isLoggedIn() ? 2 : -1), t.R7$(), t.vxM(e.isLoggedIn() ? 3 : -1), t.R7$(2), t.vxM(e.isLoggedIn() ? 5 : -1), t.R7$(), t.vxM(e.isLoggedIn() ? 6 : -1), t.R7$(), t.vxM(e.isLoggedIn() ? 7 : -1)
                }
            }

            function zn(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 64), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return i.onToggleSidenav(), s.Njj(i.logout())
                    }), t.j41(1, "mat-icon"), t.EFF(2, " power_settings_new "), t.k0s(), t.j41(3, "span", 13), t.EFF(4), t.nI1(5, "translate"), t.k0s()()
                }
                2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "TITLE_LOGOUT"), " "))
            }

            function Jn(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 65), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.onToggleSidenav())
                    }), t.j41(1, "mat-icon"), t.EFF(2, "sentiment_dissatisfied"), t.k0s(), t.j41(3, "span", 13), t.EFF(4), t.nI1(5, "translate"), t.k0s()()
                }
                2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "NAV_COMPLAIN"), " "))
            }

            function Qn(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 66), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.onToggleSidenav())
                    }), t.j41(1, "mat-icon"), t.EFF(2, "chat"), t.k0s(), t.j41(3, "span", 13), t.EFF(4), t.nI1(5, "translate"), t.k0s()()
                }
                2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "SECTION_SUPPORT_CHAT"), " "))
            }

            function qn(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 67), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.onToggleSidenav())
                    }), t.j41(1, "mat-icon"), t.EFF(2, "card_membership"), t.k0s(), t.j41(3, "span", 13), t.EFF(4), t.nI1(5, "translate"), t.k0s()()
                }
                2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "LABEL_DELUXE_MEMBERSHIP"), " "))
            }

            function Kn(n, a) {
                1 & n && t.nrm(0, "mat-divider", 19)
            }

            function Zn(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 68), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.onToggleSidenav())
                    }), t.nrm(1, "mat-icon", 69), t.j41(2, "span", 13), t.EFF(3), t.nI1(4, "translate"), t.k0s()()
                }
                2 & n && (t.R7$(3), t.SpI(" ", t.bMT(4, 1, "TITLE_SCORE_BOARD"), " "))
            }

            function to(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "a", 70), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.startHackingInstructor())
                    }), t.j41(1, "mat-icon"), t.EFF(2, "school"), t.k0s(), t.j41(3, "span", 13), t.EFF(4), t.nI1(5, "translate"), t.k0s()()
                }
                2 & n && (t.R7$(4), t.SpI(" ", t.bMT(5, 1, "BTN_GETTING_STARTED"), " "))
            }

            function eo(n, a) {
                1 & n && (t.j41(0, "a", 22), t.nrm(1, "mat-icon", 71), t.j41(2, "span", 13), t.EFF(3, " GitHub "), t.k0s()())
            }
            let no = (() => {
                class n {
                    administrationService;
                    challengeService;
                    ngZone;
                    io;
                    userService;
                    cookieService;
                    router;
                    configurationService;
                    loginGuard;
                    applicationName = "OWASP Juice Shop";
                    showGitHubLink = !0;
                    userEmail = "";
                    scoreBoardVisible = !1;
                    version = "";
                    showPrivacySubmenu = !1;
                    showOrdersSubmenu = !1;
                    isShowing = !1;
                    offerScoreBoardTutorial = !1;
                    sidenavToggle = new t.bkB;
                    constructor(e, o, i, r, c, p, k, x, O) {
                        this.administrationService = e, this.challengeService = o, this.ngZone = i, this.io = r, this.userService = c, this.cookieService = p, this.router = k, this.configurationService = x, this.loginGuard = O
                    }
                    ngOnInit() {
                        this.administrationService.getApplicationVersion().subscribe({
                            next: e => {
                                e && (this.version = `v${e}`)
                            },
                            error: e => {
                                console.log(e)
                            }
                        }), this.getApplicationDetails(), this.getScoreBoardStatus(), localStorage.getItem("token") ? this.getUserDetails() : this.userEmail = "", this.userService.getLoggedInState().subscribe(e => {
                            e ? this.getUserDetails() : this.userEmail = ""
                        }), this.ngZone.runOutsideAngular(() => {
                            this.io.socket().on("challenge solved", e => {
                                "scoreBoardChallenge" === e.key && (this.scoreBoardVisible = !0)
                            })
                        })
                    }
                    isLoggedIn() {
                        return localStorage.getItem("token")
                    }
                    logout() {
                        var e = this;
                        this.userService.saveLastLoginIp().subscribe({
                            next: o => {
                                this.noop()
                            },
                            error: o => {
                                console.log(o)
                            }
                        }), localStorage.removeItem("token"), this.cookieService.remove("token"), sessionStorage.removeItem("bid"), sessionStorage.removeItem("itemTotal"), this.userService.isLoggedIn.next(!1), this.ngZone.run((0, F.A)(function*() {
                            return yield e.router.navigate(["/"])
                        }))
                    }
                    goToProfilePage() {
                        window.location.replace(y.c.hostServer + "/profile")
                    }
                    goToDataErasurePage() {
                        window.location.replace(y.c.hostServer + "/dataerasure")
                    }
                    noop() {}
                    getScoreBoardStatus() {
                        this.challengeService.find({
                            name: "Score Board"
                        }).subscribe({
                            next: e => {
                                this.ngZone.run(() => {
                                    this.scoreBoardVisible = e[0].solved
                                })
                            },
                            error: e => {
                                console.log(e)
                            }
                        })
                    }
                    getUserDetails() {
                        this.userService.whoAmI().subscribe({
                            next: e => {
                                this.userEmail = e.email
                            },
                            error: e => {
                                console.log(e)
                            }
                        })
                    }
                    onToggleSidenav = () => {
                        this.sidenavToggle.emit()
                    };
                    getApplicationDetails() {
                        this.configurationService.getApplicationConfiguration().subscribe({
                            next: e => {
                                e?.application?.name && (this.applicationName = e.application.name), e?.application && (this.showGitHubLink = e.application.showGitHubLinks), e?.application.welcomeBanner.showOnFirstStart && e.hackingInstructor.isEnabled && (this.offerScoreBoardTutorial = e.application.welcomeBanner.showOnFirstStart && e.hackingInstructor.isEnabled)
                            },
                            error: e => {
                                console.log(e)
                            }
                        })
                    }
                    isAccounting() {
                        return "accounting" === this.loginGuard.tokenDecode()?.data?.role
                    }
                    startHackingInstructor() {
                        this.onToggleSidenav(), console.log('Starting instructions for challenge "Score Board"'), m.e(781).then(m.bind(m, 1143)).then(e => {
                            e.startHackingInstructorFor("Score Board")
                        })
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(jt), t.rXU(Z), t.rXU(t.SKi), t.rXU(et), t.rXU(P), t.rXU($.O1), t.rXU(R.Ix), t.rXU(M.s), t.rXU(X))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["sidenav"]
                        ],
                        outputs: {
                            sidenavToggle: "sidenavToggle"
                        },
                        decls: 72,
                        vars: 27,
                        consts: [
                            ["color", "primary", 1, "mat-elevation-z6"],
                            ["mat-button", "", 2, "height", "0", "position", "absolute"],
                            ["mat-subheader", "", "translate", "", 1, "side-subHeader", "show-lt-md-only"],
                            [1, "show-lt-md-only"],
                            ["mat-list-item", "", "routerLink", "/login", "aria-label", "Go to login page"],
                            ["mat-list-item", "", "aria-label", "Go to user profile"],
                            ["mat-list-item", "", "routerLink", "/accounting", "aria-label", "Go to accounting page"],
                            ["aria-label", "Show Orders and Payment Menu", 1, "parent"],
                            [1, "submenu", 3, "ngClass"],
                            ["aria-label", "Show Privacy and Security Menu", 1, "parent"],
                            ["mat-list-item", "", "aria-label", "Logout"],
                            ["mat-subheader", "", "translate", "", 1, "side-subHeader"],
                            ["mat-list-item", "", "routerLink", "/contact", "aria-label", "Go to contact us page", 3, "click"],
                            [1, "menu-text", "truncate"],
                            ["mat-list-item", "", "routerLink", "/complain", "aria-label", "Go to complain page"],
                            ["mat-list-item", "", "routerLink", "/chatbot", "aria-label", "Go to chatbot page"],
                            ["mat-list-item", "", "routerLink", "/about", "aria-label", "Go to about us page", 3, "click"],
                            ["mat-list-item", "", "routerLink", "/photo-wall", "aria-label", "Go to photo wall", 3, "click"],
                            ["mat-list-item", "", "routerLink", "/deluxe-membership", "aria-label", "Go to deluxe membership page"],
                            [2, "margin-bottom", "10px"],
                            ["mat-list-item", "", "routerLink", "/score-board", "aria-label", "Open score-board"],
                            ["mat-list-item", "", "aria-label", "Launch beginners tutorial"],
                            ["mat-list-item", "", "href", "./redirect?to=https://github.com/juice-shop/juice-shop", "aria-label", "Go to OWASP Juice Shop GitHub page"],
                            [1, "appVersion"],
                            [2, "font-size", "13px"],
                            [2, "font-size", "12px"],
                            [2, "margin-top", "10px"],
                            [1, "icon-angular"],
                            [1, "icon-html5"],
                            [1, "icon-sass"],
                            [1, "icon-css3"],
                            [1, "icon-javascript-alt"],
                            [1, "icon-nodejs"],
                            [1, "icon-database-alt2"],
                            [1, "icon-mongodb"],
                            ["mat-list-item", "", "routerLink", "/login", "aria-label", "Go to login page", 3, "click"],
                            ["mat-list-item", "", "aria-label", "Go to user profile", 3, "click"],
                            ["mat-list-item", "", "routerLink", "/accounting", "aria-label", "Go to accounting page", 3, "click"],
                            ["aria-label", "Show Orders and Payment Menu", 1, "parent", 3, "click"],
                            [1, "menu-button", 3, "ngClass"],
                            ["mat-list-item", "", "routerLink", "/order-history", "aria-label", "Go to order history page"],
                            ["mat-list-item", "", "routerLink", "/recycle", "aria-label", "Go to recycling page"],
                            ["mat-list-item", "", "routerLink", "/address/saved", "aria-label", "Go to saved address page"],
                            ["mat-list-item", "", "routerLink", "/saved-payment-methods", "aria-label", "Go to saved payment methods page"],
                            ["mat-list-item", "", "routerLink", "/wallet", "aria-label", "Go to wallet page"],
                            ["mat-list-item", "", "routerLink", "/order-history", "aria-label", "Go to order history page", 3, "click"],
                            ["mat-list-item", "", "routerLink", "/recycle", "aria-label", "Go to recycling page", 3, "click"],
                            ["mat-list-item", "", "routerLink", "/address/saved", "aria-label", "Go to saved address page", 3, "click"],
                            ["mat-list-item", "", "routerLink", "/saved-payment-methods", "aria-label", "Go to saved payment methods page", 3, "click"],
                            ["mat-list-item", "", "routerLink", "/wallet", "aria-label", "Go to wallet page", 3, "click"],
                            ["aria-label", "Show Privacy and Security Menu", 1, "parent", 3, "click"],
                            ["mat-list-item", "", "routerLink", "privacy-security/privacy-policy", "aria-label", "Go to privacy policy page"],
                            ["mat-list-item", "", "routerLink", "privacy-security/data-export", "aria-label", "Go to data export page"],
                            ["mat-list-item", "", "aria-label", "Go to data subject page"],
                            ["mat-list-item", "", "routerLink", "privacy-security/change-password", "aria-label", "Go to change password page"],
                            ["mat-list-item", "", "routerLink", "privacy-security/two-factor-authentication", "aria-label", "Go to two factor authentication page"],
                            ["mat-list-item", "", "routerLink", "privacy-security/last-login-ip", "aria-label", "Go to last login ip page"],
                            ["mat-list-item", "", "routerLink", "privacy-security/privacy-policy", "aria-label", "Go to privacy policy page", 3, "click"],
                            ["translate", "", 1, "menu-text", "truncate"],
                            ["mat-list-item", "", "routerLink", "privacy-security/data-export", "aria-label", "Go to data export page", 3, "click"],
                            ["mat-list-item", "", "aria-label", "Go to data subject page", 3, "click"],
                            ["mat-list-item", "", "routerLink", "privacy-security/change-password", "aria-label", "Go to change password page", 3, "click"],
                            ["mat-list-item", "", "routerLink", "privacy-security/two-factor-authentication", "aria-label", "Go to two factor authentication page", 3, "click"],
                            ["mat-list-item", "", "routerLink", "privacy-security/last-login-ip", "aria-label", "Go to last login ip page", 3, "click"],
                            ["mat-list-item", "", "aria-label", "Logout", 3, "click"],
                            ["mat-list-item", "", "routerLink", "/complain", "aria-label", "Go to complain page", 3, "click"],
                            ["mat-list-item", "", "routerLink", "/chatbot", "aria-label", "Go to chatbot page", 3, "click"],
                            ["mat-list-item", "", "routerLink", "/deluxe-membership", "aria-label", "Go to deluxe membership page", 3, "click"],
                            ["mat-list-item", "", "routerLink", "/score-board", "aria-label", "Open score-board", 3, "click"],
                            ["matListIcon", "", 1, "fas", "fa-trophy", "fa-lg"],
                            ["mat-list-item", "", "aria-label", "Launch beginners tutorial", 3, "click"],
                            ["matListIcon", "", 1, "fab", "fa-github", "fa-lg"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "mat-toolbar", 0)(1, "mat-toolbar-row")(2, "h2"), t.EFF(3), t.k0s()()(), t.j41(4, "mat-nav-list"), t.nrm(5, "button", 1), t.j41(6, "h3", 2), t.EFF(7, "ACCOUNT"), t.k0s(), t.j41(8, "div", 3), t.nVh(9, Mn, 6, 3, "a", 4), t.nVh(10, In, 5, 1, "a", 5), t.nVh(11, On, 6, 3, "a", 6), t.nVh(12, Pn, 8, 6, "mat-list-item", 7), t.nVh(13, Bn, 7, 8, "div", 8), t.nVh(14, Nn, 8, 6, "mat-list-item", 9), t.nVh(15, Wn, 8, 9, "div", 8), t.nVh(16, zn, 6, 3, "a", 10), t.k0s(), t.nrm(17, "mat-divider", 3), t.j41(18, "h3", 11), t.EFF(19, "TITLE_CONTACT"), t.k0s(), t.j41(20, "a", 12), t.bIt("click", function() {
                                return i.onToggleSidenav()
                            }), t.j41(21, "mat-icon"), t.EFF(22, "feedback"), t.k0s(), t.j41(23, "span", 13), t.EFF(24), t.nI1(25, "translate"), t.k0s()(), t.nVh(26, Jn, 6, 3, "a", 14), t.nVh(27, Qn, 6, 3, "a", 15), t.nrm(28, "mat-divider"), t.j41(29, "h3", 11), t.EFF(30, "COMPANY"), t.k0s(), t.j41(31, "a", 16), t.bIt("click", function() {
                                return i.onToggleSidenav()
                            }), t.j41(32, "mat-icon"), t.EFF(33, "business_center"), t.k0s(), t.j41(34, "span", 13), t.EFF(35), t.nI1(36, "translate"), t.k0s()(), t.j41(37, "a", 17), t.bIt("click", function() {
                                return i.onToggleSidenav()
                            }), t.j41(38, "mat-icon"), t.EFF(39, "camera"), t.k0s(), t.j41(40, "span", 13), t.EFF(41), t.nI1(42, "translate"), t.k0s()(), t.nVh(43, qn, 6, 3, "a", 18), t.nVh(44, Kn, 1, 0, "mat-divider", 19), t.nVh(45, Zn, 5, 3, "a", 20), t.nVh(46, to, 6, 3, "a", 21), t.nVh(47, eo, 4, 0, "a", 22), t.k0s(), t.j41(48, "div", 23)(49, "span")(50, "span", 24), t.EFF(51), t.k0s(), t.nrm(52, "br"), t.j41(53, "span", 25), t.EFF(54), t.k0s(), t.nrm(55, "br"), t.j41(56, "div", 26), t.nrm(57, "i", 27), t.EFF(58, "\xa0 "), t.nrm(59, "i", 28), t.EFF(60, "\xa0 "), t.nrm(61, "i", 29), t.EFF(62, "\xa0 "), t.nrm(63, "i", 30), t.EFF(64, "\xa0 "), t.nrm(65, "i", 31), t.EFF(66, "\xa0 "), t.nrm(67, "i", 32), t.EFF(68, "\xa0 "), t.nrm(69, "i", 33), t.EFF(70, "\xa0 "), t.nrm(71, "i", 34), t.k0s()()()), 2 & o && (t.R7$(3), t.SpI(" ", i.applicationName), t.R7$(6), t.vxM(i.isLoggedIn() ? -1 : 9), t.R7$(), t.vxM(i.isLoggedIn() ? 10 : -1), t.R7$(), t.vxM(i.isLoggedIn() && i.isAccounting() ? 11 : -1), t.R7$(), t.vxM(i.isLoggedIn() ? 12 : -1), t.R7$(), t.vxM(i.showOrdersSubmenu ? 13 : -1), t.R7$(), t.vxM(i.isLoggedIn() ? 14 : -1), t.R7$(), t.vxM(i.showPrivacySubmenu ? 15 : -1), t.R7$(), t.vxM(i.isLoggedIn() ? 16 : -1), t.R7$(8), t.SpI(" ", t.bMT(25, 21, "SECTION_CUSTOMER_FEEDBACK"), " "), t.R7$(2), t.vxM(i.isLoggedIn() ? 26 : -1), t.R7$(), t.vxM(i.isLoggedIn() ? 27 : -1), t.R7$(8), t.SpI(" ", t.bMT(36, 23, "TITLE_ABOUT"), " "), t.R7$(6), t.SpI(" ", t.bMT(42, 25, "LABEL_PHOTO_WALL"), " "), t.R7$(2), t.vxM(i.isLoggedIn() ? 43 : -1), t.R7$(), t.vxM(i.scoreBoardVisible || i.showGitHubLink ? 44 : -1), t.R7$(), t.vxM(i.scoreBoardVisible ? 45 : -1), t.R7$(), t.vxM(!i.scoreBoardVisible && i.offerScoreBoardTutorial ? 46 : -1), t.R7$(), t.vxM(i.showGitHubLink ? 47 : -1), t.R7$(4), t.JRh(i.applicationName), t.R7$(3), t.JRh(i.version))
                        },
                        dependencies: [dt.KQ, dt.uu, Ct._L, b.Hl, b.$z, Ct.EI, d.h, d.Mm, Ct.YE, nt.Wk, f.m_, f.An, Y.YU, U.q, d.D9],
                        styles: ["\n\n\n\nmat-toolbar[_ngcontent-%COMP%]{font-size:20px;height:auto;min-width:100%;padding-bottom:5px;width:100%}a[_ngcontent-%COMP%]{text-decoration:none!important}.menu-text[_ngcontent-%COMP%]{font-size:16px;padding-left:10px}.mat-mdc-list-base[_ngcontent-%COMP%], .mat-mdc-list-item[_ngcontent-%COMP%], .mat-mdc-list-item-icon[_ngcontent-%COMP%]{padding:0!important}.mat-mdc-list-item[_ngcontent-%COMP%]     .mdc-list-item__primary-text{align-items:center;display:flex;padding:0 16px}.toolbar-title[_ngcontent-%COMP%]{margin:0 16px}.mat-mdc-subheader[_ngcontent-%COMP%]{font-size:18px;font-weight:900;margin-left:18px;margin-top:10px}.app-nav-list-icon[_ngcontent-%COMP%]{margin:0 12px}.side-subHeader[_ngcontent-%COMP%]{font-weight:300}mat-nav-list[_ngcontent-%COMP%]{margin-bottom:50px}.menu-button[_ngcontent-%COMP%]{transform:rotate(0);transition:.3s ease-in-out}.menu-button.rotated[_ngcontent-%COMP%]{transform:rotate(180deg)}.submenu[_ngcontent-%COMP%]{overflow-y:hidden;padding-left:30px;transform:scaleY(0);transform-origin:top;transition:transform .3s ease}.submenu.expanded[_ngcontent-%COMP%]{transform:scaleY(1)}.appVersion[_ngcontent-%COMP%]{margin-bottom:20px;margin-top:20px;text-align:center}.truncate[_ngcontent-%COMP%]{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.show-lt-md-only[_ngcontent-%COMP%]{display:none!important}@media (max-width: 959.98px){.show-lt-md-only[_ngcontent-%COMP%]{display:block!important}}"]
                    })
                }
                return n
            })();
            var Dt = m(882);
            S.tT.watch();
            let oo = (() => {
                class n {
                    _document;
                    translate;
                    constructor(e, o) {
                        this._document = e, this.translate = o, this.translate.setDefaultLang("en")
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(s.qQL), t.rXU(d.c$))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-root"]
                        ],
                        decls: 9,
                        vars: 0,
                        consts: [
                            ["sidenav", ""],
                            ["fullscreen", ""],
                            ["mode", "over"],
                            [3, "sidenavToggle"]
                        ],
                        template: function(o, i) {
                            if (1 & o) {
                                const r = t.RV6();
                                t.j41(0, "mat-sidenav-container", 1)(1, "mat-sidenav", 2, 0)(3, "sidenav", 3), t.bIt("sidenavToggle", function() {
                                    s.eBV(r);
                                    const p = t.sdS(2);
                                    return s.Njj(p.toggle())
                                }), t.k0s()(), t.j41(4, "app-navbar", 3), t.bIt("sidenavToggle", function() {
                                    s.eBV(r);
                                    const p = t.sdS(2);
                                    return s.Njj(p.toggle())
                                }), t.k0s(), t.nrm(5, "app-server-started-notification")(6, "app-challenge-solved-notification")(7, "app-welcome")(8, "router-outlet"), t.k0s()
                            }
                        },
                        dependencies: [Dt.US, Dt.LG, no, En, an, tn, He, R.n3],
                        styles: ["\n\n\n\n.mat-sidenav[_ngcontent-%COMP%]{width:280px}"]
                    })
                }
                return n
            })();
            var io = m(850),
                ao = m(6471),
                $t = m(450),
                bt = m(6850),
                vt = m(3881),
                ro = m(1975),
                Lt = m(2885),
                so = m(9327),
                lo = m(5571),
                co = m(6156),
                Bt = m(7575),
                G = m(9454),
                kt = m(5084),
                Nt = m(2765),
                mt = m(6695),
                u = m(6192),
                mo = m(6881),
                po = m(3),
                St = m(2864),
                Ft = m(7200),
                pt = m(8288),
                Ut = m(4370),
                xt = m(107),
                uo = m(767);
            const le = n => ({
                juicycoin: n
            });
            S.Yv.add(N.Uoz, C.QnR, C.eST, C.wXH, C.q9p, H.wX, H.q9);
            let ho = (() => {
                    class n {
                        configurationService;
                        altcoinName = "Juicycoin";
                        constructor(e) {
                            this.configurationService = e
                        }
                        ngOnInit() {
                            this.configurationService.getApplicationConfiguration().subscribe({
                                next: e => {
                                    e?.application?.altcoinName && (this.altcoinName = e.application.altcoinName)
                                },
                                error: e => {
                                    console.log(e)
                                }
                            })
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(M.s))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-token-sale"]
                            ],
                            decls: 132,
                            vars: 53,
                            consts: [
                                ["appearance", "outlined"],
                                [1, "mdc-card", "container"],
                                [1, "whitepaper-container", "offer-container", "align-self-center"],
                                [3, "innerHtml"],
                                [1, "divider"],
                                ["appearance", "outlined", 1, "mat-elevation-z6"],
                                [1, "mdc-card"],
                                [2, "margin-left", "10px"],
                                [1, "whitepaper-buttons"],
                                ["href", "https://ponzico.win/ponzico.pdf", "target", "_blank", "rel", "noopener noreferrer"],
                                ["mat-raised-button", "", "color", "accent"],
                                [1, "fas", "fa-university", "fa-lg"],
                                ["href", "https://www.sec.gov/investor/alerts/ia_virtualcurrencies.pdf", "target", "_blank", "rel", "noopener noreferrer"],
                                [1, "fas", "fa-graduation-cap", "fa-lg"],
                                [1, "offer-box"],
                                [1, "text-justify"],
                                [1, "fab", "fa-bitcoin"],
                                [1, "badge"],
                                [1, "faq-container", "align-self-lt-md-center"],
                                [1, "title"],
                                [1, "fas", "fa-comments", "fa-2x"],
                                [1, "far", "fa-comment-alt", "fa-2x"],
                                [1, "far", "fa-comments", "fa-2x"],
                                [1, "fas", "fa-comment-alt", "fa-2x"],
                                [2, "margin-left", "10px", 3, "innerHtml"],
                                ["src", "assets/public/images/padding/56px.png"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "mat-card", 0)(1, "div", 1)(2, "div", 2)(3, "mat-card-header")(4, "mat-card-title"), t.EFF(5), t.nI1(6, "translate"), t.k0s(), t.nrm(7, "mat-card-subtitle", 3), t.nI1(8, "translate"), t.k0s(), t.nrm(9, "div", 4), t.j41(10, "mat-card", 5)(11, "div", 6)(12, "h4"), t.EFF(13), t.nI1(14, "translate"), t.j41(15, "small", 7), t.EFF(16, "("), t.j41(17, "span"), t.EFF(18), t.nI1(19, "translate"), t.k0s(), t.EFF(20, ")"), t.k0s()(), t.j41(21, "div", 8)(22, "a", 9)(23, "button", 10), t.nrm(24, "i", 11), t.EFF(25, " PonzICO Whitepaper"), t.k0s()(), t.j41(26, "a", 12)(27, "button", 10), t.nrm(28, "i", 13), t.EFF(29, " PonziCoin Whitepaper"), t.k0s()()()()(), t.j41(30, "div", 14)(31, "h3"), t.EFF(32), t.nI1(33, "translate"), t.k0s(), t.j41(34, "p", 15), t.EFF(35, " Lorem ipsum dolor sit amet "), t.j41(36, "strong"), t.nrm(37, "i", 16), t.EFF(38), t.k0s(), t.EFF(39, ", consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. "), t.j41(40, "span"), t.EFF(41), t.nI1(42, "translate"), t.k0s(), t.EFF(43, " Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat. "), t.j41(44, "span"), t.EFF(45), t.nI1(46, "translate"), t.k0s(), t.EFF(47, " Ut wisi enim ad minim veniam, quis "), t.j41(48, "strong"), t.nrm(49, "i", 16), t.EFF(50), t.k0s(), t.EFF(51, " nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat. Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi. "), t.j41(52, "span"), t.EFF(53), t.nI1(54, "translate"), t.k0s(), t.EFF(55, " Nam liber tempor cum soluta nobis eleifend option congue nihil imperdiet doming id quod mazim placerat facer possim assum. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat. Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat. "), t.j41(56, "span"), t.EFF(57), t.nI1(58, "translate"), t.k0s(), t.EFF(59, " Duis autem vel eum "), t.j41(60, "strong"), t.nrm(61, "i", 16), t.EFF(62), t.k0s(), t.EFF(63, " iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis. "), t.j41(64, "span"), t.EFF(65), t.nI1(66, "translate"), t.k0s(), t.EFF(67, " At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, At accusam aliquyam diam diam dolore dolores duo eirmod eos erat, et nonumy sed tempor et et invidunt justo labore Stet clita ea et gubergren, kasd magna no rebum. sanctus sea sed takimata ut vero voluptua. "), t.j41(68, "span", 17), t.nrm(69, "i", 16), t.EFF(70), t.k0s(), t.EFF(71, " est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur. "), t.k0s()()(), t.j41(72, "div", 18)(73, "mat-card-header")(74, "mat-card-title", 19), t.EFF(75), t.nI1(76, "translate"), t.k0s()(), t.nrm(77, "div", 4), t.j41(78, "mat-card", 5)(79, "div", 6)(80, "h5"), t.nrm(81, "i", 20), t.EFF(82, " Stet "), t.j41(83, "strong"), t.nrm(84, "i", 16), t.EFF(85), t.k0s(), t.EFF(86, " clita kasd gubergren?"), t.k0s(), t.j41(87, "small", 15), t.EFF(88, " Stet clita kasd gubergren, no "), t.j41(89, "strong"), t.nrm(90, "i", 16), t.EFF(91), t.k0s(), t.EFF(92, " sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. "), t.k0s()()(), t.j41(93, "mat-card", 5)(94, "div", 6)(95, "h5"), t.nrm(96, "i", 21), t.EFF(97, " Consetetur sadipscing elitr?"), t.k0s(), t.j41(98, "small", 15), t.EFF(99, " Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed "), t.j41(100, "strong"), t.nrm(101, "i", 16), t.EFF(102), t.k0s(), t.EFF(103, " diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. "), t.k0s()()(), t.j41(104, "mat-card", 5)(105, "div", 6)(106, "h5"), t.nrm(107, "i", 22), t.EFF(108, " Hendrerit "), t.j41(109, "strong"), t.nrm(110, "i", 16), t.EFF(111), t.k0s(), t.EFF(112, " in vulputate velit?"), t.k0s(), t.j41(113, "small", 15), t.EFF(114, " Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis. "), t.k0s()()(), t.j41(115, "mat-card", 5)(116, "div", 6)(117, "h5"), t.nrm(118, "i", 23), t.EFF(119, " Justo duo dolores et ea rebum?"), t.k0s(), t.j41(120, "small", 15), t.EFF(121, " At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. "), t.k0s()()(), t.j41(122, "mat-card", 5)(123, "div", 6)(124, "h5"), t.nrm(125, "i", 20)(126, "span", 24), t.nI1(127, "translate"), t.k0s(), t.j41(128, "small", 15), t.EFF(129), t.nI1(130, "translate"), t.k0s()()()(), t.nrm(131, "img", 25), t.k0s()()), 2 & o && (t.R7$(5), t.SpI(" ", t.bMT(6, 21, "TITLE_TOKENSALE"), " "), t.R7$(2), t.Y8G("innerHtml", t.i5U(8, 23, "SECTION_ICO", t.eq3(49, le, i.altcoinName)), t.npT), t.R7$(6), t.SpI("", t.bMT(14, 26, "SECTION_WHITEPAPER"), " "), t.R7$(5), t.JRh(t.bMT(19, 28, "WHITEPAPER_REFERENCES")), t.R7$(14), t.JRh(t.bMT(33, 30, "SECTION_SALES_PITCH")), t.R7$(6), t.SpI(" ", i.altcoinName), t.R7$(3), t.JRh(t.bMT(42, 32, "GIVE_US_ALL_YOUR_MONEY")), t.R7$(4), t.JRh(t.bMT(46, 34, "GIVE_US_ALL_YOUR_MONEY")), t.R7$(5), t.SpI(" ", i.altcoinName), t.R7$(3), t.JRh(t.bMT(54, 36, "GIVE_US_ALL_YOUR_MONEY")), t.R7$(4), t.JRh(t.bMT(58, 38, "GIVE_US_ALL_YOUR_MONEY")), t.R7$(5), t.SpI(" ", i.altcoinName), t.R7$(3), t.JRh(t.bMT(66, 40, "GIVE_US_ALL_YOUR_MONEY")), t.R7$(5), t.SpI(" ", i.altcoinName), t.R7$(5), t.JRh(t.bMT(76, 42, "ICO_FAQ")), t.R7$(10), t.SpI(" ", i.altcoinName), t.R7$(6), t.SpI(" ", i.altcoinName), t.R7$(11), t.SpI(" ", i.altcoinName), t.R7$(9), t.SpI(" ", i.altcoinName), t.R7$(15), t.Y8G("innerHtml", t.i5U(127, 44, "ICO_FAQ_QUESTION", t.eq3(51, le, i.altcoinName)), t.npT), t.R7$(3), t.JRh(t.bMT(130, 47, "ICO_FAQ_ANSWER")))
                            },
                            dependencies: [h.Hu, h.RN, h.MM, h.Lc, h.dh, d.h, b.Hl, b.$z, d.D9],
                            styles: ["\n\n\n\n.container[_ngcontent-%COMP%]{align-items:stretch;display:flex;flex-direction:row;gap:20px;justify-content:center}@media (max-width: 959.98px){.container[_ngcontent-%COMP%]{flex-direction:column}.align-self-lt-md-center[_ngcontent-%COMP%]{align-self:center}}.align-self-center[_ngcontent-%COMP%]{align-self:center}.heading[_ngcontent-%COMP%]{background:#00000021;justify-content:center;margin-bottom:10px;padding:12px 20px}.whitepaper-container.offer-container[_ngcontent-%COMP%]{max-width:700px;min-width:300px;width:70%}.whitepaper-buttons[_ngcontent-%COMP%]{display:flex;flex-direction:row;flex-wrap:wrap;gap:8px}.faq-container[_ngcontent-%COMP%]{max-width:500px;min-width:200px;width:70%}a[_ngcontent-%COMP%]:nth-child(1){padding-right:10px}.title[_ngcontent-%COMP%]{padding-bottom:27px}.divider[_ngcontent-%COMP%]{border:.5px solid #f2f2f2;margin-bottom:10px}.text-justify[_ngcontent-%COMP%]{text-align:justify}mat-card[_ngcontent-%COMP%]{margin-bottom:10px}mat-card[_ngcontent-%COMP%]:nth-child(1){border-radius:5px;margin:0 auto;width:80%}mat-card-title[_ngcontent-%COMP%]{font-size:20px}mat-card-header[_ngcontent-%COMP%]{margin-left:-15px;margin-top:15px}h4[_ngcontent-%COMP%]{font-size:15px}.faq-container[_ngcontent-%COMP%]   mat-card-title[_ngcontent-%COMP%]{margin-bottom:12px}img[_ngcontent-%COMP%]{width:10px}.mdc-card[_ngcontent-%COMP%]{border:0}.offer-container[_ngcontent-%COMP%]{display:flex;flex-direction:column;gap:16px}.faq-container[_ngcontent-%COMP%]{display:flex;flex-direction:column;gap:4px;margin-top:-7px}"]
                        })
                    }
                    return n
                })(),
                go = (() => {
                    class n {
                        cookieService;
                        userService;
                        router;
                        route;
                        ngZone;
                        constructor(e, o, i, r, c) {
                            this.cookieService = e, this.userService = o, this.router = i, this.route = r, this.ngZone = c
                        }
                        ngOnInit() {
                            var e = this;
                            this.userService.oauthLogin(this.parseRedirectUrlParams().access_token).subscribe({
                                next: o => {
                                    const i = btoa(o.email.split("").reverse().join(""));
                                    this.userService.save({
                                        email: o.email,
                                        password: i,
                                        passwordRepeat: i
                                    }).subscribe({
                                        next: () => {
                                            this.login(o)
                                        },
                                        error: () => {
                                            this.login(o)
                                        }
                                    })
                                },
                                error: o => {
                                    this.invalidateSession(o), this.ngZone.run((0, F.A)(function*() {
                                        return yield e.router.navigate(["/login"])
                                    }))
                                }
                            })
                        }
                        login(e) {
                            var o = this;
                            this.userService.login({
                                email: e.email,
                                password: btoa(e.email.split("").reverse().join("")),
                                oauth: !0
                            }).subscribe({
                                next: i => {
                                    const r = new Date;
                                    r.setHours(r.getHours() + 8), this.cookieService.put("token", i.token, {
                                        expires: r
                                    }), localStorage.setItem("token", i.token), sessionStorage.setItem("bid", i.bid), this.userService.isLoggedIn.next(!0), this.ngZone.run((0, F.A)(function*() {
                                        return yield o.router.navigate(["/"])
                                    }))
                                },
                                error: i => {
                                    this.invalidateSession(i), this.ngZone.run((0, F.A)(function*() {
                                        return yield o.router.navigate(["/login"])
                                    }))
                                }
                            })
                        }
                        invalidateSession(e) {
                            console.log(e), this.cookieService.remove("token"), localStorage.removeItem("token"), sessionStorage.removeItem("bid")
                        }
                        parseRedirectUrlParams() {
                            const o = this.route.snapshot.data.params.substr(1).split("&"),
                                i = {};
                            for (let r = 0; r < o.length; r++) {
                                const c = o[r].split("=");
                                i[c[0]] = c[1]
                            }
                            return i
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU($.O1), t.rXU(P), t.rXU(R.Ix), t.rXU(R.nX), t.rXU(t.SKi))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-oauth"]
                            ],
                            decls: 9,
                            vars: 6,
                            consts: [
                                [1, "center-container"],
                                ["appearance", "outlined", 1, "primary-notification"],
                                [1, "mdc-card"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "div", 0)(1, "mat-card", 1)(2, "div", 2)(3, "h3"), t.EFF(4), t.nI1(5, "translate"), t.k0s(), t.j41(6, "div"), t.EFF(7), t.nI1(8, "translate"), t.k0s()()()()), 2 & o && (t.R7$(4), t.JRh(t.bMT(5, 2, "TITLE_LOGIN")), t.R7$(3), t.JRh(t.bMT(8, 4, "CONFIRM_LOGGED_IN_VIA_OAUTH2")))
                            },
                            dependencies: [h.Hu, h.RN, d.h, d.D9],
                            styles: ["\n\n\n\nmat-card[_ngcontent-%COMP%]{height:auto;min-width:300px;width:35%}.center-container[_ngcontent-%COMP%]{align-items:stretch;display:flex;justify-content:center}"]
                        })
                    }
                    return n
                })();
            var L = m(8559);

            function fo(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 15)
            }

            function _o(n, a) {
                if (1 & n && (t.j41(0, "mat-cell", 16), t.nrm(1, "img", 17), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.Y8G("alt", t.mNQ(e.name))("src", "assets/public/images/products/" + e.image, t.B4B)
                }
            }

            function Co(n, a) {
                1 & n && t.nrm(0, "mat-footer-cell", 16)
            }

            function bo(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 15)
            }

            function vo(n, a) {
                if (1 & n && (t.j41(0, "mat-cell", 18), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.SpI(" ", e.name)
                }
            }

            function ko(n, a) {
                1 & n && t.nrm(0, "mat-footer-cell")
            }

            function So(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 15)
            }

            function Fo(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 20), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG().$implicit,
                            r = t.XpG();
                        return s.Njj(r.dec(i.BasketItem.id))
                    }), t.nrm(1, "i", 21), t.k0s()
                }
            }

            function xo(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 20), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG().$implicit,
                            r = t.XpG();
                        return s.Njj(r.inc(i.BasketItem.id))
                    }), t.nrm(1, "i", 22), t.k0s()
                }
            }

            function To(n, a) {
                if (1 & n && (t.j41(0, "mat-cell", 16), t.nVh(1, Fo, 2, 0, "button", 19), t.j41(2, "span", 18), t.EFF(3), t.k0s(), t.nVh(4, xo, 2, 0, "button", 19), t.k0s()), 2 & n) {
                    const e = a.$implicit,
                        o = t.XpG();
                    t.R7$(), t.vxM(o.allowEdit ? 1 : -1), t.R7$(2), t.SpI(" ", e.BasketItem.quantity), t.R7$(), t.vxM(o.allowEdit ? 4 : -1)
                }
            }

            function yo(n, a) {
                1 & n && (t.j41(0, "mat-footer-cell", 23), t.EFF(1, "TOTAL"), t.k0s())
            }

            function Ro(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 15)
            }

            function wo(n, a) {
                if (1 & n && (t.j41(0, "mat-cell", 18), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.SpI(" ", e.price, "\xa4")
                }
            }

            function Eo(n, a) {
                1 & n && t.nrm(0, "mat-footer-cell")
            }

            function Mo(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 15)
            }

            function Io(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-cell")(1, "button", 20), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG(2);
                        return s.Njj(r.delete(i.BasketItem.id))
                    }), t.nrm(2, "i", 25), t.k0s()()
                }
            }

            function Oo(n, a) {
                1 & n && t.nrm(0, "mat-footer-cell")
            }

            function Po(n, a) {
                1 & n && (t.qex(0, 11), t.DNE(1, Mo, 1, 0, "mat-header-cell", 2)(2, Io, 3, 0, "mat-cell", 24)(3, Oo, 1, 0, "mat-footer-cell", 7), t.bVm())
            }

            function jo(n, a) {
                1 & n && t.nrm(0, "mat-header-row")
            }

            function Ao(n, a) {
                1 & n && t.nrm(0, "mat-row")
            }

            function Do(n, a) {
                1 & n && t.nrm(0, "mat-footer-row", 27)
            }

            function $o(n, a) {
                if (1 & n && (t.j41(0, "span"), t.DNE(1, Do, 1, 0, "mat-footer-row", 26), t.k0s()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(), t.Y8G("matFooterRowDef", e.tableColumns)
                }
            }

            function Lo(n, a) {
                if (1 & n && (t.j41(0, "div", 14), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(), t.Lme("", t.bMT(2, 2, "LABEL_TOTAL_PRICE"), ": ", e.itemTotal, "\xa4")
                }
            }
            S.Yv.add(H.UB, C.xbI, C.jkA);
            let ce = (() => {
                class n {
                    deluxeGuard;
                    basketService;
                    userService;
                    snackBarHelperService;
                    allowEdit = !1;
                    displayTotal = !1;
                    totalPrice = !0;
                    emitTotal = new t.bkB;
                    emitProductCount = new t.bkB;
                    tableColumns = ["image", "product", "quantity", "price"];
                    dataSource = [];
                    bonus = 0;
                    itemTotal = 0;
                    userEmail;
                    constructor(e, o, i, r) {
                        this.deluxeGuard = e, this.basketService = o, this.userService = i, this.snackBarHelperService = r
                    }
                    ngOnInit() {
                        this.allowEdit && !this.tableColumns.includes("remove") && this.tableColumns.push("remove"), this.load(), this.userService.whoAmI().subscribe({
                            next: e => {
                                this.userEmail = e.email || "anonymous", this.userEmail = "(" + this.userEmail + ")"
                            },
                            error: e => {
                                console.log(e)
                            }
                        })
                    }
                    load() {
                        this.basketService.find(parseInt(sessionStorage.getItem("bid"), 10)).subscribe({
                            next: e => {
                                this.isDeluxe() && e.Products.forEach(o => {
                                    o.price = o.deluxePrice
                                }), this.dataSource = e.Products, this.itemTotal = e.Products.reduce((o, i) => o + i.price * i.BasketItem.quantity, 0), this.bonus = e.Products.reduce((o, i) => o + Math.round(i.price / 10) * i.BasketItem.quantity, 0), this.sendToParent(this.dataSource.length)
                            },
                            error: e => {
                                console.log(e)
                            }
                        })
                    }
                    delete(e) {
                        this.basketService.del(e).subscribe({
                            next: () => {
                                this.load(), this.basketService.updateNumberOfCartItems()
                            },
                            error: o => {
                                console.log(o)
                            }
                        })
                    }
                    inc(e) {
                        this.addToQuantity(e, 1)
                    }
                    dec(e) {
                        this.addToQuantity(e, -1)
                    }
                    addToQuantity(e, o) {
                        this.basketService.get(e).subscribe({
                            next: i => {
                                const r = i.quantity + o;
                                this.basketService.put(e, {
                                    quantity: r < 1 ? 1 : r
                                }).subscribe({
                                    next: () => {
                                        this.load(), this.basketService.updateNumberOfCartItems()
                                    },
                                    error: c => {
                                        this.snackBarHelperService.open(c.error?.error, "errorBar"), console.log(c)
                                    }
                                })
                            },
                            error: i => {
                                console.log(i)
                            }
                        })
                    }
                    sendToParent(e) {
                        this.emitTotal.emit([this.itemTotal, this.bonus]), this.emitProductCount.emit(e)
                    }
                    isDeluxe() {
                        return this.deluxeGuard.isDeluxe()
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(At), t.rXU(Q), t.rXU(P), t.rXU(L.r))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-purchase-basket"]
                        ],
                        inputs: {
                            allowEdit: "allowEdit",
                            displayTotal: "displayTotal",
                            totalPrice: "totalPrice"
                        },
                        outputs: {
                            emitTotal: "emitTotal",
                            emitProductCount: "emitProductCount"
                        },
                        decls: 27,
                        vars: 10,
                        consts: [
                            [3, "dataSource"],
                            ["matColumnDef", "image"],
                            ["style", "display: none;", 4, "matHeaderCellDef"],
                            ["class", "content-align", 4, "matCellDef"],
                            ["class", "content-align", 4, "matFooterCellDef"],
                            ["matColumnDef", "product"],
                            ["style", "font-size: initial;", 4, "matCellDef"],
                            [4, "matFooterCellDef"],
                            ["matColumnDef", "quantity"],
                            ["class", "header-align", 4, "matFooterCellDef"],
                            ["matColumnDef", "price"],
                            ["matColumnDef", "remove"],
                            [4, "matHeaderRowDef"],
                            [4, "matRowDef", "matRowDefColumns"],
                            ["id", "price"],
                            [2, "display", "none"],
                            [1, "content-align"],
                            [1, "img-responsive", "img-thumbnail", 3, "src", "alt"],
                            [2, "font-size", "initial"],
                            ["mat-icon-button", ""],
                            ["mat-icon-button", "", 3, "click"],
                            [1, "fas", "fa-minus-square"],
                            [1, "fas", "fa-plus-square"],
                            [1, "header-align"],
                            [4, "matCellDef"],
                            [1, "far", "fa-trash-alt"],
                            ["mat-footer-row", "", 4, "matFooterRowDef"],
                            ["mat-footer-row", ""]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "h1"), t.EFF(1), t.nI1(2, "translate"), t.j41(3, "small"), t.EFF(4), t.k0s()(), t.j41(5, "mat-table", 0), t.qex(6, 1), t.DNE(7, fo, 1, 0, "mat-header-cell", 2)(8, _o, 2, 3, "mat-cell", 3)(9, Co, 1, 0, "mat-footer-cell", 4), t.bVm(), t.qex(10, 5), t.DNE(11, bo, 1, 0, "mat-header-cell", 2)(12, vo, 2, 1, "mat-cell", 6)(13, ko, 1, 0, "mat-footer-cell", 7), t.bVm(), t.qex(14, 8), t.DNE(15, So, 1, 0, "mat-header-cell", 2)(16, To, 5, 3, "mat-cell", 3)(17, yo, 2, 0, "mat-footer-cell", 9), t.bVm(), t.qex(18, 10), t.DNE(19, Ro, 1, 0, "mat-header-cell", 2)(20, wo, 2, 1, "mat-cell", 6)(21, Eo, 1, 0, "mat-footer-cell", 7), t.bVm(), t.nVh(22, Po, 4, 0, "ng-container", 11), t.DNE(23, jo, 1, 0, "mat-header-row", 12)(24, Ao, 1, 0, "mat-row", 13), t.nVh(25, $o, 2, 1, "span"), t.k0s(), t.nVh(26, Lo, 3, 4, "div", 14)), 2 & o && (t.R7$(), t.SpI(" ", t.bMT(2, 8, "TITLE_BASKET"), " "), t.R7$(3), t.JRh(i.userEmail), t.R7$(), t.Y8G("dataSource", i.dataSource), t.R7$(17), t.vxM(i.allowEdit ? 22 : -1), t.R7$(), t.Y8G("matHeaderRowDef", i.tableColumns), t.R7$(), t.Y8G("matRowDefColumns", i.tableColumns), t.R7$(), t.vxM(i.displayTotal ? 25 : -1), t.R7$(), t.vxM(i.totalPrice ? 26 : -1))
                        },
                        dependencies: [u.Zl, u.cC, u.tL, u.KS, u.YV, u.$R, u.Zq, u.Qo, V.iY, u.ji, u.YZ, u.iL, u.NB, u.xW, u.iF, d.h, d.D9],
                        styles: ["\n\n\n\nh1[_ngcontent-%COMP%]{margin-bottom:24px}h1[_ngcontent-%COMP%]   small[_ngcontent-%COMP%]{color:#0009;font-size:.8em;margin-left:8px}.heading[_ngcontent-%COMP%]{background:#0003;border-radius:4px;font-size:x-large;justify-content:center;margin-bottom:20px;padding:12px 20px}mat-table[_ngcontent-%COMP%]{margin-bottom:24px;table-layout:fixed;width:100%}mat-header-row[_ngcontent-%COMP%]{min-height:0!important}mat-row[_ngcontent-%COMP%]{transition:background-color .2s ease}mat-row[_ngcontent-%COMP%]:hover{background-color:#0000000a}mat-cell[_ngcontent-%COMP%], mat-footer-cell[_ngcontent-%COMP%]{padding:16px}mat-cell[_ngcontent-%COMP%]:first-of-type, mat-footer-cell[_ngcontent-%COMP%]:first-of-type{padding-left:16px}mat-cell[_ngcontent-%COMP%]:last-of-type, mat-footer-cell[_ngcontent-%COMP%]:last-of-type{padding-right:16px}.mat-column-image[_ngcontent-%COMP%]{width:20%}.mat-column-product[_ngcontent-%COMP%]{width:35%}.mat-column-quantity[_ngcontent-%COMP%], .mat-column-price[_ngcontent-%COMP%]{width:16%}.mat-column-remove[_ngcontent-%COMP%]{width:10%}@media (max-width: 959.98px){.mat-column-image[_ngcontent-%COMP%], .mat-column-price[_ngcontent-%COMP%]{display:none}.mat-column-product[_ngcontent-%COMP%]{width:30%}.mat-column-quantity[_ngcontent-%COMP%]{width:35%}.mat-column-remove[_ngcontent-%COMP%]{width:15%}}.img-thumbnail[_ngcontent-%COMP%]{border-radius:4px;height:auto;max-width:90px;transition:transform .2s ease}.img-thumbnail[_ngcontent-%COMP%]:hover{transform:scale(1.05)}.mat-mdc-footer-cell[_ngcontent-%COMP%]{font-size:1.1em;font-weight:500}.content-align[_ngcontent-%COMP%]{align-items:center;display:flex;justify-content:center}.quantity-controls[_ngcontent-%COMP%]{align-items:center;display:flex;gap:8px}.quantity-controls[_ngcontent-%COMP%]   button[_ngcontent-%COMP%]{color:#0009;transition:color .2s ease}.quantity-controls[_ngcontent-%COMP%]   button[_ngcontent-%COMP%]:hover{color:#000000de}.quantity-controls[_ngcontent-%COMP%]   span[_ngcontent-%COMP%]{min-width:24px;text-align:center}#price[_ngcontent-%COMP%]{font-size:large;font-weight:500;margin-top:30px;padding:16px;text-align:right;width:100%}@media (max-width: 768px){.img-thumbnail[_ngcontent-%COMP%]{max-width:70px}mat-cell[_ngcontent-%COMP%], mat-footer-cell[_ngcontent-%COMP%]{padding:12px}#price[_ngcontent-%COMP%]{font-size:medium;margin-top:20px}}.far.fa-trash-alt[_ngcontent-%COMP%]{color:#0000008a;transition:color .2s ease}.far.fa-trash-alt[_ngcontent-%COMP%]:hover{color:#f44336}"]
                    })
                }
                return n
            })();
            const Bo = n => ({
                bonus: n
            });
            S.Yv.add(C.Tu5);
            let No = (() => {
                    class n {
                        router;
                        ngZone;
                        productCount = 0;
                        bonus = 0;
                        constructor(e, o) {
                            this.router = e, this.ngZone = o
                        }
                        checkout() {
                            var e = this;
                            this.ngZone.run((0, F.A)(function*() {
                                return yield e.router.navigate(["/address/select"])
                            }))
                        }
                        getProductCount(e) {
                            this.productCount = e
                        }
                        getBonusPoints(e) {
                            sessionStorage.setItem("itemTotal", e[0]), this.bonus = e[1]
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(R.Ix), t.rXU(t.SKi))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-basket"]
                            ],
                            decls: 9,
                            vars: 8,
                            consts: [
                                ["appearance", "outlined", 1, "mat-elevation-z6"],
                                [1, "mdc-card"],
                                [3, "emitTotal", "emitProductCount", "allowEdit"],
                                ["id", "checkoutButton", "color", "primary", "mat-raised-button", "", 1, "checkout-button", 3, "click", "disabled"],
                                [1, "fas", "fa-cart-arrow-down"],
                                ["translate", "", 1, "hint", 3, "translateParams"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "mat-card", 0)(1, "div", 1)(2, "app-purchase-basket", 2), t.bIt("emitTotal", function(c) {
                                    return i.getBonusPoints(c)
                                })("emitProductCount", function(c) {
                                    return i.getProductCount(c)
                                }), t.k0s(), t.j41(3, "button", 3), t.bIt("click", function() {
                                    return i.checkout()
                                }), t.nrm(4, "i", 4), t.EFF(5), t.nI1(6, "translate"), t.k0s(), t.j41(7, "div", 5), t.EFF(8, "CHECKOUT_FOR_BONUS_POINTS"), t.k0s()()()), 2 & o && (t.R7$(2), t.Y8G("allowEdit", !0), t.R7$(), t.Y8G("disabled", i.productCount < 1), t.R7$(2), t.SpI(" ", t.bMT(6, 4, "BTN_CHECKOUT"), " "), t.R7$(2), t.Y8G("translateParams", t.eq3(6, Bo, i.bonus)))
                            },
                            dependencies: [h.Hu, h.RN, ce, b.Hl, b.$z, d.h, d.Mm, d.D9],
                            styles: ["\n\n\n\n.container[_ngcontent-%COMP%]{min-width:420px;width:40%}mat-card[_ngcontent-%COMP%]{margin-left:20%;margin-right:20%}.heading[_ngcontent-%COMP%]{background:#00000021;justify-content:center;padding:12px 20px}.mat-mdc-row[_ngcontent-%COMP%]{padding-bottom:10px}.mat-column-description[_ngcontent-%COMP%], .mat-column-bonus[_ngcontent-%COMP%], .mat-column-price[_ngcontent-%COMP%]{margin-left:20px}mat-form-field[_ngcontent-%COMP%]{margin-top:15px;width:100%}mat-expansion-panel[_ngcontent-%COMP%]{margin-bottom:20px}#checkoutButton[_ngcontent-%COMP%]{margin-left:20%;margin-top:30px;width:60%}.hint[_ngcontent-%COMP%]{display:flex;font-size:13px;justify-content:center;margin-top:5px}.payment-label[_ngcontent-%COMP%]{padding-top:10px}"]
                        })
                    }
                    return n
                })(),
                Vt = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/rest/track-order";
                        constructor(e) {
                            this.http = e
                        }
                        find(e) {
                            return e = encodeURIComponent(e), this.http.get(`${this.host}/${e}`).pipe((0, v.T)(o => o), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })();
            var it = m(345);
            const Uo = n => ({
                bonus: n
            });

            function Vo(n, a) {
                1 & n && (t.j41(0, "span"), t.nrm(1, "i", 26), t.k0s())
            }

            function Go(n, a) {
                1 & n && (t.j41(0, "span"), t.nrm(1, "i", 27), t.k0s())
            }

            function Yo(n, a) {
                1 & n && (t.j41(0, "span"), t.nrm(1, "i", 28), t.k0s())
            }

            function Ho(n, a) {
                1 & n && (t.j41(0, "span"), t.nrm(1, "i", 29), t.k0s())
            }

            function Xo(n, a) {
                1 & n && (t.j41(0, "span"), t.nrm(1, "i", 30), t.k0s())
            }

            function Wo(n, a) {
                1 & n && (t.j41(0, "span"), t.nrm(1, "i", 31), t.k0s())
            }

            function zo(n, a) {
                if (1 & n && (t.j41(0, "span", 8)(1, "span"), t.nrm(2, "i", 32), t.k0s(), t.j41(3, "span", 33), t.EFF(4), t.nI1(5, "translate"), t.k0s()()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(4), t.Lme("", e.results.eta, " ", t.bMT(5, 2, "LABEL_DAYS"))
                }
            }

            function Jo(n, a) {
                if (1 & n && (t.j41(0, "span", 8)(1, "span"), t.nrm(2, "i", 34), t.k0s(), t.j41(3, "span", 33), t.EFF(4), t.nI1(5, "translate"), t.k0s()()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(4), t.Lme("", e.results.eta, " ", t.bMT(5, 2, "LABEL_DAYS"))
                }
            }

            function Qo(n, a) {
                1 & n && (t.j41(0, "mat-header-cell", 3), t.EFF(1, "LABEL_PRODUCT"), t.k0s())
            }

            function qo(n, a) {
                if (1 & n && (t.j41(0, "mat-cell", 35), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.SpI(" ", e.name)
                }
            }

            function Ko(n, a) {
                1 & n && (t.j41(0, "mat-header-cell", 3), t.EFF(1, "LABEL_PRICE"), t.k0s())
            }

            function Zo(n, a) {
                if (1 & n && (t.j41(0, "mat-cell", 36), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.SpI(" ", e.price, "\xa4")
                }
            }

            function ti(n, a) {
                1 & n && (t.j41(0, "mat-header-cell", 3), t.EFF(1, "LABEL_QUANTITY"), t.k0s())
            }

            function ei(n, a) {
                if (1 & n && (t.j41(0, "mat-cell", 37), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.SpI(" ", e.quantity)
                }
            }

            function ni(n, a) {
                1 & n && (t.j41(0, "mat-header-cell", 3), t.EFF(1, "LABEL_TOTAL_PRICE"), t.k0s())
            }

            function oi(n, a) {
                if (1 & n && (t.j41(0, "mat-cell", 38), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.SpI(" ", e.total, "\xa4")
                }
            }

            function ii(n, a) {
                1 & n && t.nrm(0, "mat-header-row")
            }

            function ai(n, a) {
                1 & n && t.nrm(0, "mat-row")
            }
            S.Yv.add(C.D_b, C.VNe, C.ZMv, C.raf, C.v02);
            var at = function(n) {
                return n[n.New = 0] = "New", n[n.Packing = 1] = "Packing", n[n.Transit = 2] = "Transit", n[n.Delivered = 3] = "Delivered", n
            }(at || {});
            let de = (() => {
                    class n {
                        route;
                        trackOrderService;
                        sanitizer;
                        displayedColumns = ["product", "price", "quantity", "total price"];
                        dataSource = new u.I6;
                        orderId;
                        results = {};
                        status = at.New;
                        Status = at;
                        constructor(e, o, i) {
                            this.route = e, this.trackOrderService = o, this.sanitizer = i
                        }
                        ngOnInit() {
                            this.orderId = this.route.snapshot.queryParams.id, this.trackOrderService.find(this.orderId).subscribe(e => {
                                this.results.orderNo = this.sanitizer.bypassSecurityTrustHtml(`<code>${e.data[0].orderId}</code>`), this.results.email = e.data[0].email, this.results.totalPrice = e.data[0].totalPrice, this.results.products = e.data[0].products, this.results.eta = void 0 !== e.data[0].eta ? e.data[0].eta : "?", this.results.bonus = e.data[0].bonus, this.dataSource.data = this.results.products, this.status = e.data[0].delivered ? at.Delivered : this.route.snapshot.data.type ? at.New : this.results.eta > 2 ? at.Packing : at.Transit
                            })
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(R.nX), t.rXU(Vt), t.rXU(it.up))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-track-result"]
                            ],
                            decls: 49,
                            vars: 14,
                            consts: [
                                ["table", ""],
                                ["appearance", "outlined", 1, "mat-elevation-z6", "mat-own-card"],
                                [1, "mdc-card"],
                                ["translate", ""],
                                [3, "innerHtml"],
                                [2, "text-align", "center"],
                                [1, "container-fluid", "well"],
                                [1, "row", "fa-4x"],
                                [1, "fa-layers", "fa-fw"],
                                [1, "table-wrapper"],
                                [1, "table-heading", "heading"],
                                [1, "mdc-table"],
                                [1, "mat-elevation-z0", 3, "dataSource"],
                                ["matColumnDef", "product"],
                                ["translate", "", 4, "matHeaderCellDef"],
                                ["class", "product-name", 4, "matCellDef"],
                                ["matColumnDef", "price"],
                                ["class", "product-price", 4, "matCellDef"],
                                ["matColumnDef", "quantity"],
                                ["class", "product-quantity", 4, "matCellDef"],
                                ["matColumnDef", "total price"],
                                ["class", "product-total", 4, "matCellDef"],
                                [4, "matHeaderRowDef"],
                                [4, "matRowDef", "matRowDefColumns"],
                                [1, "bonus-container"],
                                ["translate", "", 3, "translateParams"],
                                [1, "fas", "fa-warehouse", "confirmation"],
                                [1, "fas", "fa-warehouse"],
                                [1, "fas", "fa-truck-loading", "confirmation"],
                                [1, "fas", "fa-truck-loading"],
                                [1, "fas", "fa-truck", "confirmation"],
                                [1, "fas", "fa-truck"],
                                [1, "fas", "fa-home", "confirmation"],
                                [1, "fa-layers-counter", "accent-notification", 2, "width", "max-content"],
                                [1, "fas", "fa-home"],
                                [1, "product-name"],
                                [1, "product-price"],
                                [1, "product-quantity"],
                                [1, "product-total"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "mat-card", 1)(1, "div", 2)(2, "h1")(3, "span", 3), t.EFF(4, "TITLE_SEARCH_RESULTS"), t.k0s(), t.EFF(5, " - "), t.nrm(6, "span", 4), t.k0s(), t.j41(7, "div", 5)(8, "h3", 3), t.EFF(9, "LABEL_EXPECTED_DELIVERY"), t.k0s(), t.j41(10, "div", 6)(11, "div", 7), t.nVh(12, Vo, 2, 0, "span")(13, Go, 2, 0, "span"), t.nVh(14, Yo, 2, 0, "span")(15, Ho, 2, 0, "span"), t.nVh(16, Xo, 2, 0, "span")(17, Wo, 2, 0, "span"), t.nVh(18, zo, 6, 4, "span", 8)(19, Jo, 6, 4, "span", 8), t.k0s()()(), t.j41(20, "div", 9)(21, "div", 10)(22, "h2", 3), t.EFF(23, "LABEL_PRODUCT_ORDERED"), t.k0s()()(), t.j41(24, "div", 11)(25, "mat-table", 12, 0), t.qex(27, 13), t.DNE(28, Qo, 2, 0, "mat-header-cell", 14)(29, qo, 2, 1, "mat-cell", 15), t.bVm(), t.qex(30, 16), t.DNE(31, Ko, 2, 0, "mat-header-cell", 14)(32, Zo, 2, 1, "mat-cell", 17), t.bVm(), t.qex(33, 18), t.DNE(34, ti, 2, 0, "mat-header-cell", 14)(35, ei, 2, 1, "mat-cell", 19), t.bVm(), t.qex(36, 20), t.DNE(37, ni, 2, 0, "mat-header-cell", 14)(38, oi, 2, 1, "mat-cell", 21), t.bVm(), t.DNE(39, ii, 1, 0, "mat-header-row", 22)(40, ai, 1, 0, "mat-row", 23), t.k0s()(), t.j41(41, "div", 24)(42, "h2", 25), t.EFF(43, "BONUS_POINTS_EARNED"), t.k0s(), t.j41(44, "p"), t.EFF(45, "("), t.nrm(46, "span", 4), t.nI1(47, "translate"), t.EFF(48, ")"), t.k0s()()()()), 2 & o && (t.R7$(6), t.Y8G("innerHtml", i.results.orderNo, t.npT), t.R7$(6), t.vxM(i.status === i.Status.New ? 12 : 13), t.R7$(2), t.vxM(i.status === i.Status.Packing ? 14 : 15), t.R7$(2), t.vxM(i.status === i.Status.Transit ? 16 : 17), t.R7$(2), t.vxM(i.status === i.Status.Delivered ? 18 : 19), t.R7$(7), t.Y8G("dataSource", i.dataSource), t.R7$(14), t.Y8G("matHeaderRowDef", i.displayedColumns), t.R7$(), t.Y8G("matRowDefColumns", i.displayedColumns), t.R7$(2), t.Y8G("translateParams", t.eq3(12, Uo, i.results.bonus)), t.R7$(4), t.Y8G("innerHtml", t.bMT(47, 10, "BONUS_FOR_FUTURE_PURCHASES"), t.npT))
                            },
                            dependencies: [h.Hu, h.RN, d.h, d.Mm, u.Zl, u.cC, u.tL, u.KS, u.YV, u.$R, u.ji, u.YZ, u.iL, u.NB, d.D9],
                            styles: ["\n\n\n\nmat-card[_ngcontent-%COMP%]   h1[_ngcontent-%COMP%]{margin-top:0}.mdc-table[_ngcontent-%COMP%]{align-self:center;margin:0 auto;width:90%}.heading[_ngcontent-%COMP%]{background:#0003;height:35px;justify-content:center;padding:0 20px;width:90%}.product-name[_ngcontent-%COMP%]{margin-right:25px}.table-wrapper[_ngcontent-%COMP%]{display:flex;justify-content:center;margin-top:30px}.table-heading[_ngcontent-%COMP%]   h4[_ngcontent-%COMP%]{margin-bottom:0}.row.fa-4x[_ngcontent-%COMP%]   svg[_ngcontent-%COMP%]{margin-right:10px}.mat-own-card[_ngcontent-%COMP%]{display:block;margin-left:20%;margin-right:20%}.bonus-container[_ngcontent-%COMP%]{align-items:center;box-sizing:border-box;display:flex;flex-direction:column;margin-top:10px;place-content:stretch center}.bonus-container[_ngcontent-%COMP%]   h2[_ngcontent-%COMP%]{margin:0;padding:0}@media (max-width: 959.98px){.mat-column-price[_ngcontent-%COMP%]{display:none}}"]
                        })
                    }
                    return n
                })(),
                Tt = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/api/Feedbacks";
                        constructor(e) {
                            this.http = e
                        }
                        find(e) {
                            return this.http.get(this.host + "/", {
                                params: e
                            }).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        save(e) {
                            return this.http.post(this.host + "/", e).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        del(e) {
                            return this.http.delete(`${this.host}/${e}`).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })(),
                me = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/rest/captcha";
                        constructor(e) {
                            this.http = e
                        }
                        getCaptcha() {
                            return this.http.get(this.host + "/").pipe((0, g.W)(e => {
                                throw e
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })(),
                tt = (() => {
                    class n {
                        _document;
                        constructor(e) {
                            this._document = e
                        }
                        attachEnterKeyHandler(e, o, i) {
                            const r = this._document.getElementById(e),
                                c = this._document.getElementById(o);
                            r.addEventListener("keyup", function(p) {
                                p.preventDefault(), 13 === p.keyCode && !c.disabled && i()
                            })
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(s.qQL))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })();
            const ri = () => ({
                length: "160"
            });

            function si(n, a) {
                1 & n && (t.j41(0, "mat-error", 4), t.EFF(1, "MANDATORY_COMMENT "), t.k0s())
            }

            function li(n, a) {
                1 & n && (t.j41(0, "mat-error", 4), t.EFF(1, "MANDATORY_CAPTCHA "), t.k0s())
            }

            function ci(n, a) {
                1 & n && (t.j41(0, "mat-error", 4), t.EFF(1, "INVALID_CAPTCHA "), t.k0s())
            }
            S.Yv.add(C.yy, C.isI);
            let di = (() => {
                class n {
                    userService;
                    captchaService;
                    feedbackService;
                    formSubmitService;
                    translate;
                    snackBarHelperService;
                    authorControl = new l.hs({
                        value: "",
                        disabled: !0
                    }, []);
                    feedbackControl = new l.hs("", [l.k0.required, l.k0.maxLength(160)]);
                    captchaControl = new l.hs("", [l.k0.required, l.k0.pattern("-?[\\d]*")]);
                    userIdControl = new l.hs("", []);
                    rating = 0;
                    feedback = void 0;
                    captcha;
                    captchaId;
                    confirmation;
                    error;
                    constructor(e, o, i, r, c, p) {
                        this.userService = e, this.captchaService = o, this.feedbackService = i, this.formSubmitService = r, this.translate = c, this.snackBarHelperService = p
                    }
                    ngOnInit() {
                        this.userService.whoAmI().subscribe({
                            next: e => {
                                this.feedback = {}, this.userIdControl.setValue(e.id), this.feedback.UserId = e.id, this.authorControl.setValue(e.email ? `***${e.email.slice(3)}` : "anonymous")
                            },
                            error: e => {
                                this.feedback = void 0, console.log(e)
                            }
                        }), this.getNewCaptcha(), this.formSubmitService.attachEnterKeyHandler("feedback-form", "submitButton", () => {
                            this.save()
                        })
                    }
                    getNewCaptcha() {
                        this.captchaService.getCaptcha().subscribe({
                            next: e => {
                                this.captcha = e.captcha, this.captchaId = e.captchaId
                            },
                            error: e => e
                        })
                    }
                    save() {
                        this.feedback.captchaId = this.captchaId, this.feedback.captcha = this.captchaControl.value, this.feedback.comment = `${this.feedbackControl.value} (${this.authorControl.value})`, this.feedback.rating = this.rating, this.feedback.UserId = this.userIdControl.value, this.feedbackService.save(this.feedback).subscribe({
                            next: e => {
                                5 === e.rating ? this.translate.get("FEEDBACK_FIVE_STAR_THANK_YOU").subscribe({
                                    next: o => {
                                        this.snackBarHelperService.open(o)
                                    },
                                    error: o => {
                                        this.snackBarHelperService.open(o)
                                    }
                                }) : this.translate.get("FEEDBACK_THANK_YOU").subscribe({
                                    next: o => {
                                        this.snackBarHelperService.open(o)
                                    },
                                    error: o => {
                                        this.snackBarHelperService.open(o)
                                    }
                                }), this.feedback = {}, this.ngOnInit(), this.resetForm()
                            },
                            error: e => {
                                console.log(e), this.snackBarHelperService.open(e.error, "errorBar"), this.feedback = {}, this.resetCaptcha()
                            }
                        })
                    }
                    resetForm() {
                        this.authorControl.markAsUntouched(), this.authorControl.markAsPristine(), this.authorControl.setValue(""), this.feedbackControl.markAsUntouched(), this.feedbackControl.markAsPristine(), this.feedbackControl.setValue(""), this.rating = 0, this.captchaControl.markAsUntouched(), this.captchaControl.markAsPristine(), this.captchaControl.setValue("")
                    }
                    resetCaptcha() {
                        this.captchaControl.markAsUntouched(), this.captchaControl.markAsPristine(), this.captchaControl.setValue("")
                    }
                    formatRating(e) {
                        return `${e}\u2605`
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(P), t.rXU(me), t.rXU(Tt), t.rXU(tt), t.rXU(d.c$), t.rXU(L.r))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-contact"]
                        ],
                        decls: 54,
                        vars: 28,
                        consts: [
                            ["comment", ""],
                            [1, "center-container"],
                            ["appearance", "outlined", 1, "mat-elevation-z6"],
                            [1, "mdc-card"],
                            ["translate", ""],
                            ["id", "feedback-form", 1, "form-container"],
                            ["hidden", "", "type", "text", "id", "userId", "ngDefaultControl", "", 3, "formControl"],
                            ["appearance", "outline", "color", "accent"],
                            ["ngDefaultControl", "", "matInput", "", "type", "text", "aria-label", "Field with the name of the author", 3, "formControl"],
                            [1, "fas", "fa-exclamation-circle"],
                            ["translate", "", 2, "margin-left", "5px"],
                            ["id", "comment", "ngDefaultControl", "", "matInput", "", "matAutosizeMinRows", "4", "matAutosizeMaxRows", "4", "matTextareaAutosize", "", "cols", "50", "maxlength", "160", "aria-label", "Field for entering the comment or the feedback", 3, "formControl", "placeholder"],
                            ["align", "end"],
                            [1, "rating-container"],
                            ["translate", "", 2, "font-weight", "500", "margin-right", "8px", "float", "left"],
                            ["id", "rating", "min", "1", "max", "5", "step", "1", "showTickMarks", "", "discrete", "", "aria-label", "Slider for selecting the star rating", 3, "displayWith"],
                            ["matSliderThumb", "", 3, "ngModelChange", "ngModel"],
                            [2, "margin-bottom", "10px", "margin-top", "10px"],
                            [2, "font-weight", "500"],
                            ["translate", "", 2, "font-size", "small"],
                            ["id", "captcha", "aria-label", "CAPTCHA code which must be solved"],
                            [2, "font-size", "small"],
                            ["id", "captchaControl", "ngDefaultControl", "", "matInput", "", "type", "text", "aria-label", "Field for the result of the CAPTCHA code", "pattern", "-?[\\d]*", 3, "formControl", "placeholder"],
                            ["type", "submit", "id", "submitButton", "mat-raised-button", "", "color", "primary", "aria-label", "Button to send the review", 3, "click", "disabled"]
                        ],
                        template: function(o, i) {
                            if (1 & o) {
                                const r = t.RV6();
                                t.j41(0, "div", 1)(1, "mat-card", 2)(2, "div", 3)(3, "h1", 4), t.EFF(4, "SECTION_CUSTOMER_FEEDBACK"), t.k0s(), t.j41(5, "div", 5), t.nrm(6, "input", 6), t.j41(7, "mat-form-field", 7)(8, "mat-label", 4), t.EFF(9, "LABEL_AUTHOR"), t.k0s(), t.nrm(10, "input", 8), t.k0s(), t.j41(11, "mat-form-field", 7)(12, "mat-label", 4), t.EFF(13, "LABEL_COMMENT"), t.k0s(), t.j41(14, "mat-hint", 4), t.nrm(15, "i", 9), t.j41(16, "em", 10), t.EFF(17), t.nI1(18, "translate"), t.k0s()(), t.nrm(19, "textarea", 11, 0), t.nI1(21, "translate"), t.j41(22, "mat-hint", 12), t.EFF(23), t.k0s(), t.nVh(24, si, 2, 0, "mat-error", 4), t.k0s(), t.j41(25, "div", 13)(26, "label", 14), t.EFF(27, "LABEL_RATING"), t.k0s(), t.j41(28, "mat-slider", 15)(29, "input", 16), t.mxI("ngModelChange", function(p) {
                                    return s.eBV(r), t.DH7(i.rating, p) || (i.rating = p), s.Njj(p)
                                }), t.k0s()()(), t.j41(30, "div", 17)(31, "label", 18), t.EFF(32, "CAPTCHA:"), t.k0s(), t.EFF(33, "\xa0\xa0"), t.j41(34, "span", 19), t.EFF(35, "LABEL_WHAT_IS"), t.k0s(), t.EFF(36, "\xa0 "), t.j41(37, "code", 20), t.EFF(38), t.k0s(), t.EFF(39, "\xa0"), t.j41(40, "label", 21), t.EFF(41, "?"), t.k0s()(), t.j41(42, "mat-form-field", 7)(43, "mat-label", 4), t.EFF(44, "LABEL_RESULT"), t.k0s(), t.nrm(45, "input", 22), t.nI1(46, "translate"), t.nVh(47, li, 2, 0, "mat-error", 4), t.nVh(48, ci, 2, 0, "mat-error", 4), t.k0s()(), t.j41(49, "button", 23), t.bIt("click", function() {
                                    return s.eBV(r), s.Njj(i.save())
                                }), t.j41(50, "mat-icon"), t.EFF(51, "send"), t.k0s(), t.EFF(52), t.nI1(53, "translate"), t.k0s()()()()
                            }
                            if (2 & o) {
                                const r = t.sdS(20);
                                t.R7$(6), t.Y8G("formControl", i.userIdControl), t.R7$(4), t.Y8G("formControl", i.authorControl), t.R7$(7), t.JRh(t.i5U(18, 18, "MAX_TEXTAREA_LENGTH", t.lJ4(27, ri))), t.R7$(2), t.Y8G("placeholder", t.mNQ(t.bMT(21, 21, "WRITE_REVIEW_PLACEHOLDER")))("formControl", i.feedbackControl), t.R7$(4), t.SpI("", (null == r.value ? null : r.value.length) || 0, "/160"), t.R7$(), t.vxM(i.feedbackControl.invalid && i.feedbackControl.errors.required ? 24 : -1), t.R7$(4), t.Y8G("displayWith", i.formatRating), t.R7$(), t.R50("ngModel", i.rating), t.R7$(9), t.JRh(i.captcha), t.R7$(7), t.Y8G("placeholder", t.mNQ(t.bMT(46, 23, "MANDATORY_CAPTCHA")))("formControl", i.captchaControl), t.R7$(2), t.vxM(i.captchaControl.invalid && i.captchaControl.errors.required ? 47 : -1), t.R7$(), t.vxM(i.captchaControl.invalid && i.captchaControl.errors.pattern ? 48 : -1), t.R7$(), t.Y8G("disabled", i.authorControl.invalid || i.feedbackControl.invalid || i.captchaControl.invalid || !i.rating), t.R7$(3), t.SpI(" ", t.bMT(53, 25, "BTN_SUBMIT"), " ")
                            }
                        },
                        dependencies: [h.Hu, h.RN, d.h, d.Mm, l.YN, l.me, l.BC, l.tU, l.R_, l.vS, l.X1, l.l_, I.R, _.rl, _.nJ, _.MV, _.TL, T.fS, T.fg, vt.Ez, vt.IV, vt.OU, b.Hl, b.$z, f.m_, f.An, d.D9],
                        styles: ["\n\n\n\nmat-card[_ngcontent-%COMP%]{height:auto;min-width:320px;width:30%}.form-container[_ngcontent-%COMP%]{display:flex;flex-direction:column;position:relative}.rating-container[_ngcontent-%COMP%]{align-items:center;display:flex;margin-bottom:15px;margin-top:15px}  .mat-mdc-slider{--mdc-slider-handle-color: var(--theme-accent);--mdc-slider-focus-handle-color: var(--theme-accent);--mdc-slider-hover-handle-color: var(--theme-accent);--mdc-slider-active-track-color: var(--theme-accent);--mdc-slider-inactive-track-color: var(--theme-accent-fade-30);--mdc-slider-with-tick-marks-active-container-color: var(--theme-background);--mdc-slider-with-tick-marks-inactive-container-color: var(--theme-background-darker)}  .mat-mdc-slider .mdc-slider__thumb-knob{background-color:var(--theme-accent)!important;border-color:var(--theme-accent)!important}  .mat-mdc-slider .mdc-slider__tick-marks{background-size:25% 2px!important}  .mat-mdc-slider .mdc-slider__thumb-label{background-color:var(--theme-accent);color:var(--theme-background)}mat-hint[_ngcontent-%COMP%]{color:var(--theme-text-fade-40);font-size:.85em}.star[_ngcontent-%COMP%]{color:#c8c8c8}.active[_ngcontent-%COMP%], .select[_ngcontent-%COMP%]{color:gold}[_nghost-%COMP%]     .br{margin:0!important}#submitButton[_ngcontent-%COMP%]{margin-left:20%;margin-top:30px;width:60%}.center-container[_ngcontent-%COMP%]{align-items:stretch;display:flex;justify-content:center}"]
                    })
                }
                return n
            })();
            var yt = m(983);

            function mi(n, a) {
                if (1 & n && (t.j41(0, "div", 8), t.nrm(1, "div", 9), t.k0s()), 2 & n) {
                    const e = t.XpG().$implicit;
                    t.R7$(), t.Y8G("innerHTML", null == e ? null : e.args, t.npT)
                }
            }

            function pi(n, a) {
                if (1 & n && (t.qex(0), t.nVh(1, mi, 2, 1, "div", 8), t.bVm()), 2 & n) {
                    const e = a.active;
                    t.R7$(), t.vxM(e ? 1 : -1)
                }
            }

            function ui(n, a) {
                if (1 & n && (t.j41(0, "a", 10)(1, "button", 18), t.nrm(2, "i", 19), t.EFF(3, " BlueSky"), t.k0s()()), 2 & n) {
                    const e = t.XpG(2);
                    t.Y8G("href", e.blueSkyUrl, t.B4B)
                }
            }

            function hi(n, a) {
                if (1 & n && (t.j41(0, "a", 11)(1, "button", 18), t.nrm(2, "i", 20), t.EFF(3, " Mastodon"), t.k0s()()), 2 & n) {
                    const e = t.XpG(2);
                    t.Y8G("href", e.mastodonUrl, t.B4B)
                }
            }

            function gi(n, a) {
                if (1 & n && (t.j41(0, "a", 12)(1, "button", 18), t.nrm(2, "i", 21), t.EFF(3, " Twitter"), t.k0s()()), 2 & n) {
                    const e = t.XpG(2);
                    t.Y8G("href", e.twitterUrl, t.B4B)
                }
            }

            function fi(n, a) {
                if (1 & n && (t.j41(0, "a", 13)(1, "button", 18), t.nrm(2, "i", 22), t.EFF(3, " Facebook"), t.k0s()()), 2 & n) {
                    const e = t.XpG(2);
                    t.Y8G("href", e.facebookUrl, t.B4B)
                }
            }

            function _i(n, a) {
                if (1 & n && (t.j41(0, "a", 14)(1, "button", 18), t.nrm(2, "i", 23), t.EFF(3, " Slack"), t.k0s()()), 2 & n) {
                    const e = t.XpG(2);
                    t.Y8G("href", e.slackUrl, t.B4B)
                }
            }

            function Ci(n, a) {
                if (1 & n && (t.j41(0, "a", 15)(1, "button", 18), t.nrm(2, "i", 24), t.EFF(3, " Reddit"), t.k0s()()), 2 & n) {
                    const e = t.XpG(2);
                    t.Y8G("href", e.redditUrl, t.B4B)
                }
            }

            function bi(n, a) {
                if (1 & n && (t.j41(0, "a", 16)(1, "button", 18), t.nrm(2, "i", 25), t.EFF(3, " Press Kit"), t.k0s()()), 2 & n) {
                    const e = t.XpG(2);
                    t.Y8G("href", e.pressKitUrl, t.B4B)
                }
            }

            function vi(n, a) {
                if (1 & n && (t.j41(0, "a", 17)(1, "button", 18), t.nrm(2, "i", 26), t.EFF(3, " NFT"), t.k0s()()), 2 & n) {
                    const e = t.XpG(2);
                    t.Y8G("href", e.nftUrl, t.B4B)
                }
            }

            function ki(n, a) {
                if (1 & n && (t.j41(0, "div", 7)(1, "h3")(2, "span"), t.EFF(3), t.nI1(4, "translate"), t.k0s()(), t.nVh(5, ui, 4, 1, "a", 10), t.nVh(6, hi, 4, 1, "a", 11), t.nVh(7, gi, 4, 1, "a", 12), t.nVh(8, fi, 4, 1, "a", 13), t.nVh(9, _i, 4, 1, "a", 14), t.nVh(10, Ci, 4, 1, "a", 15), t.nVh(11, bi, 4, 1, "a", 16), t.nVh(12, vi, 4, 1, "a", 17), t.k0s()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(3), t.JRh(t.bMT(4, 9, "SECTION_SOCIAL_MEDIA")), t.R7$(2), t.vxM(e.blueSkyUrl ? 5 : -1), t.R7$(), t.vxM(e.mastodonUrl ? 6 : -1), t.R7$(), t.vxM(e.twitterUrl ? 7 : -1), t.R7$(), t.vxM(e.facebookUrl ? 8 : -1), t.R7$(), t.vxM(e.slackUrl ? 9 : -1), t.R7$(), t.vxM(e.redditUrl ? 10 : -1), t.R7$(), t.vxM(e.pressKitUrl ? 11 : -1), t.R7$(), t.vxM(e.nftUrl ? 12 : -1)
                }
            }
            S.Yv.add(N.aUl, N.HQ1, N.qu9, N.onr, H.bG, H.yy, C.yy, C.fyG, N.TOJ, C.l5I);
            let Si = (() => {
                    class n {
                        configurationService;
                        feedbackService;
                        sanitizer;
                        gallery;
                        blueSkyUrl;
                        mastodonUrl;
                        twitterUrl;
                        facebookUrl;
                        slackUrl;
                        redditUrl;
                        pressKitUrl;
                        nftUrl;
                        galleryRef;
                        images = ["assets/public/images/carousel/1.jpg", "assets/public/images/carousel/2.jpg", "assets/public/images/carousel/3.jpg", "assets/public/images/carousel/4.jpg", "assets/public/images/carousel/5.png", "assets/public/images/carousel/6.jpg", "assets/public/images/carousel/7.jpg"];
                        stars = [null, '<i class="fas fa-star"></i><i class="far fa-star"></i><i class="far fa-star"></i><i class="far fa-star"></i><i class="far fa-star"></i>', '<i class="fas fa-star"></i><i class="fas fa-star"></i><i class="far fa-star"></i><i class="far fa-star"></i><i class="far fa-star"></i>', '<i class="fas fa-star"></i><i class="fas fa-star"></i><i class="fas fa-star"></i><i class="far fa-star"></i><i class="far fa-star"></i>', '<i class="fas fa-star"></i><i class="fas fa-star"></i><i class="fas fa-star"></i><i class="fas fa-star"></i><i class="far fa-star"></i>', '<i class="fas fa-star"></i><i class="fas fa-star"></i><i class="fas fa-star"></i><i class="fas fa-star"></i><i class="fas fa-star"></i>'];
                        constructor(e, o, i, r) {
                            this.configurationService = e, this.feedbackService = o, this.sanitizer = i, this.gallery = r
                        }
                        ngOnInit() {
                            this.galleryRef = this.gallery.ref("feedback-gallery"), this.populateSlideshowFromFeedbacks(), this.configurationService.getApplicationConfiguration().pipe((0, g.W)(e => (console.error(e), yt.w))).subscribe(e => {
                                e?.application?.social && (e.application.social.blueSkyUrl && (this.blueSkyUrl = e.application.social.blueSkyUrl), e.application.social.mastodonUrl && (this.mastodonUrl = e.application.social.mastodonUrl), e.application.social.twitterUrl && (this.twitterUrl = e.application.social.twitterUrl), e.application.social.facebookUrl && (this.facebookUrl = e.application.social.facebookUrl), e.application.social.slackUrl && (this.slackUrl = e.application.social.slackUrl), e.application.social.redditUrl && (this.redditUrl = e.application.social.redditUrl), e.application.social.pressKitUrl && (this.pressKitUrl = e.application.social.pressKitUrl), e.application.social.nftUrl && (this.nftUrl = e.application.social.nftUrl))
                            })
                        }
                        populateSlideshowFromFeedbacks() {
                            this.feedbackService.find().pipe((0, g.W)(e => (console.error(e), yt.w))).subscribe(e => {
                                for (let o = 0; o < e.length; o++) e[o].comment = `<span style="width: 90%; display:block;">${e[o].comment}<br/> (${this.stars[e[o].rating]})</span>`, e[o].comment = this.sanitizer.bypassSecurityTrustHtml(e[o].comment), this.galleryRef.addImage({
                                    src: this.images[o % this.images.length],
                                    args: e[o].comment
                                })
                            })
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(M.s), t.rXU(Tt), t.rXU(it.up), t.rXU(xt.Z2))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-about"]
                            ],
                            decls: 21,
                            vars: 14,
                            consts: [
                                ["appearance", "outlined", 1, "mat-elevation-z6"],
                                [1, "mdc-card"],
                                [1, "about-us"],
                                [1, "text-justify"],
                                ["href", "ftp/legal.md", "aria-label", "Link to the Terms of Use", "translate", ""],
                                ["id", "feedback-gallery", 2, "height", "300px", 3, "autoplay", "thumbs", "counter", "imageSize"],
                                [4, "galleryImageDef"],
                                [1, "social"],
                                [1, "item-text"],
                                [1, "caption", 3, "innerHTML"],
                                ["target", "_blank", "rel", "noopener noreferrer", "aria-label", "Button for the BlueSky page of the shop", 2, "margin-left", "0px", 3, "href"],
                                ["target", "_blank", "rel", "noopener noreferrer", "aria-label", "Button for the Mastodon page of the shop", 2, "margin-left", "0px", 3, "href"],
                                ["target", "_blank", "rel", "noopener noreferrer", "aria-label", "Button for the Twitter page of the shop", 2, "margin-left", "0px", 3, "href"],
                                ["target", "_blank", "rel", "noopener noreferrer", "aria-label", "Button for the Facebook page of the shop", 3, "href"],
                                ["target", "_blank", "rel", "noopener noreferrer", "aria-label", "Button for the Slack page of the shop", 3, "href"],
                                ["target", "_blank", "rel", "noopener noreferrer", "aria-label", "Button for the Reddit page of the shop", 3, "href"],
                                ["target", "_blank", "rel", "noopener noreferrer", "aria-label", "Button for the PressKit page of the shop", 3, "href"],
                                ["target", "_blank", "rel", "noopener noreferrer", "aria-label", "Button for the NFT of the shop", 3, "href"],
                                ["mat-raised-button", "", "color", "accent"],
                                [1, "fas", "fa-bold", "fa-lg"],
                                [1, "fab", "fa-mastodon", "fa-lg"],
                                [1, "fab", "fa-twitter", "fa-lg"],
                                [1, "fab", "fa-facebook", "fa-lg"],
                                [1, "fab", "fa-slack", "fa-lg"],
                                [1, "fab", "fa-reddit", "fa-lg"],
                                [1, "far", "fa-newspaper", "fa-lg"],
                                [1, "fas", "fa-palette", "fa-lg"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "mat-card", 0)(1, "div", 1)(2, "section", 2)(3, "h1"), t.EFF(4), t.nI1(5, "translate"), t.k0s(), t.j41(6, "h3"), t.EFF(7), t.nI1(8, "translate"), t.k0s(), t.j41(9, "p", 3), t.EFF(10, " Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat. "), t.j41(11, "a", 4), t.EFF(12, "LINK_TERMS_OF_USE"), t.k0s(), t.EFF(13, " At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, At accusam aliquyam diam diam dolore dolores duo eirmod eos erat, et nonumy sed tempor et et invidunt justo labore Stet clita ea et gubergren, kasd magna no rebum. "), t.k0s(), t.j41(14, "h3")(15, "span"), t.EFF(16), t.nI1(17, "translate"), t.k0s()(), t.j41(18, "gallery", 5), t.DNE(19, pi, 2, 1, "ng-container", 6), t.k0s(), t.nVh(20, ki, 13, 11, "div", 7), t.k0s()()()), 2 & o && (t.R7$(4), t.JRh(t.bMT(5, 8, "TITLE_ABOUT")), t.R7$(3), t.JRh(t.bMT(8, 10, "SECTION_CORPORATE_HISTORY")), t.R7$(9), t.JRh(t.bMT(17, 12, "SECTION_CUSTOMER_FEEDBACK")), t.R7$(2), t.Y8G("autoplay", !0)("thumbs", !1)("counter", !1)("imageSize", "cover"), t.R7$(2), t.vxM(i.blueSkyUrl || i.mastodonUrl || i.twitterUrl || i.facebookUrl || i.slackUrl || i.redditUrl || i.pressKitUrl || i.nftUrl ? 20 : -1))
                            },
                            dependencies: [h.Hu, h.RN, d.h, d.Mm, xt.A9, xt.t8, b.Hl, b.$z, d.D9],
                            styles: ["\n\n\n\nmat-card[_ngcontent-%COMP%]{margin-bottom:30px;margin-left:auto;margin-right:auto;width:80%}.about-us[_ngcontent-%COMP%]{margin-left:8.33333%;margin-right:8.33333%;width:83.3333%}.social[_ngcontent-%COMP%]{margin-top:20px}button[_ngcontent-%COMP%]{margin:5px}.text-justify[_ngcontent-%COMP%]{text-align:justify;text-justify:inter-word}.caption[_ngcontent-%COMP%]{background-color:#00000059;bottom:0;color:#fff;left:0;padding:10px;position:absolute;width:100%}.mdc-card[_ngcontent-%COMP%]{border:0}"]
                        })
                    }
                    return n
                })(),
                pe = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/api/SecurityAnswers";
                        constructor(e) {
                            this.http = e
                        }
                        save(e) {
                            return this.http.post(this.host + "/", e).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })(),
                Gt = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/api/SecurityQuestions";
                        constructor(e) {
                            this.http = e
                        }
                        find(e) {
                            return this.http.get(this.host + "/", {
                                params: e
                            }).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        findBy(e) {
                            return this.http.get(this.hostServer + "/rest/user/security-question?email=" + e).pipe((0, v.T)(o => o.question), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })();
            var ue = m(3029);
            let he = (() => {
                class n {
                    password = "";
                    passwordStrength = 0;
                    minLength = 8;
                    ranges = ["low", "low-medium", "medium", "high-medium", "high"];
                    ngOnChanges(e) {
                        e.password && (this.passwordStrength = this.calculatePasswordStrength(this.password))
                    }
                    calculatePasswordStrength(e) {
                        const o = [this.containAtLeastMinChars, this.containAtLeastOneLowerCaseLetter, this.containAtLeastOneUpperCaseLetter, this.containAtLeastOneDigit, this.containAtLeastOneSpecialChar];
                        return o.filter(i => i).length / o.length * 100
                    }
                    get progressColor() {
                        return this.ranges[Math.max(Math.floor(this.passwordStrength / (100 / this.ranges.length)) - 1, 0)]
                    }
                    get containAtLeastMinChars() {
                        return this.password.length >= 8
                    }
                    get containAtLeastOneLowerCaseLetter() {
                        return /^(?=.*?[a-z])/.test(this.password)
                    }
                    get containAtLeastOneUpperCaseLetter() {
                        return /^(?=.*?[A-Z])/.test(this.password)
                    }
                    get containAtLeastOneDigit() {
                        return /^(?=.*?[0-9])/.test(this.password)
                    }
                    get containAtLeastOneSpecialChar() {
                        return /^(?=.*?[" !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"])/.test(this.password)
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-password-strength"]
                        ],
                        inputs: {
                            password: "password"
                        },
                        features: [t.OA$],
                        decls: 1,
                        vars: 3,
                        consts: [
                            ["mode", "determinate", 3, "hidden", "value", "ngClass"]
                        ],
                        template: function(o, i) {
                            1 & o && t.nrm(0, "mat-progress-bar", 0), 2 & o && t.Y8G("hidden", 0 === i.password.length)("value", i.passwordStrength)("ngClass", i.progressColor)
                        },
                        dependencies: [Bt.PO, Bt.HM, Y.YU],
                        styles: [".low[_ngcontent-%COMP%]{--mdc-linear-progress-active-indicator-color: #f00 !important}.low-medium[_ngcontent-%COMP%]{--mdc-linear-progress-active-indicator-color: #ff4500 !important}.medium[_ngcontent-%COMP%]{--mdc-linear-progress-active-indicator-color: #ff0 !important}.high-medium[_ngcontent-%COMP%]{--mdc-linear-progress-active-indicator-color: #9acd32 !important}.high[_ngcontent-%COMP%]{--mdc-linear-progress-active-indicator-color: #0f0 !important}"]
                    })
                }
                return n
            })();

            function Fi(n, a) {
                1 & n && (t.j41(0, "mat-icon", 2), t.EFF(1, "done"), t.k0s())
            }

            function xi(n, a) {
                1 & n && (t.j41(0, "mat-icon", 3), t.EFF(1, "error"), t.k0s())
            }

            function Ti(n, a) {
                1 & n && (t.j41(0, "mat-icon", 2), t.EFF(1, "done"), t.k0s())
            }

            function yi(n, a) {
                1 & n && (t.j41(0, "mat-icon", 3), t.EFF(1, "error"), t.k0s())
            }

            function Ri(n, a) {
                1 & n && (t.j41(0, "mat-icon", 2), t.EFF(1, "done"), t.k0s())
            }

            function wi(n, a) {
                1 & n && (t.j41(0, "mat-icon", 3), t.EFF(1, "error"), t.k0s())
            }

            function Ei(n, a) {
                1 & n && (t.j41(0, "mat-icon", 2), t.EFF(1, "done"), t.k0s())
            }

            function Mi(n, a) {
                1 & n && (t.j41(0, "mat-icon", 3), t.EFF(1, "error"), t.k0s())
            }

            function Ii(n, a) {
                1 & n && (t.j41(0, "mat-icon", 2), t.EFF(1, "done"), t.k0s())
            }

            function Oi(n, a) {
                1 & n && (t.j41(0, "mat-icon", 3), t.EFF(1, "error"), t.k0s())
            }
            let ge = (() => {
                class n {
                    passwordComponent;
                    enableScoreInfo = !1;
                    lowerCaseCriteriaMsg = "contains at least one lower character";
                    upperCaseCriteriaMsg = "contains at least one upper character";
                    digitsCriteriaMsg = "contains at least one digit character";
                    specialCharsCriteriaMsg = "contains at least one special character";
                    minCharsCriteriaMsg;
                    ngOnInit() {
                        this.minCharsCriteriaMsg || (this.minCharsCriteriaMsg = `contains at least ${this.passwordComponent.minLength} characters`)
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-password-strength-info"]
                        ],
                        inputs: {
                            passwordComponent: "passwordComponent",
                            enableScoreInfo: "enableScoreInfo",
                            lowerCaseCriteriaMsg: "lowerCaseCriteriaMsg",
                            upperCaseCriteriaMsg: "upperCaseCriteriaMsg",
                            digitsCriteriaMsg: "digitsCriteriaMsg",
                            specialCharsCriteriaMsg: "specialCharsCriteriaMsg",
                            minCharsCriteriaMsg: "minCharsCriteriaMsg"
                        },
                        decls: 27,
                        vars: 10,
                        consts: [
                            [1, "info-card"],
                            [1, "info-row"],
                            ["color", "primary", "fontIcon", "done"],
                            ["color", "warn", "fontIcon", "error"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "mat-card", 0)(1, "mat-card-content")(2, "div", 1), t.nVh(3, Fi, 2, 0, "mat-icon", 2)(4, xi, 2, 0, "mat-icon", 3), t.j41(5, "span"), t.EFF(6), t.k0s()(), t.j41(7, "div", 1), t.nVh(8, Ti, 2, 0, "mat-icon", 2)(9, yi, 2, 0, "mat-icon", 3), t.j41(10, "span"), t.EFF(11), t.k0s()(), t.j41(12, "div", 1), t.nVh(13, Ri, 2, 0, "mat-icon", 2)(14, wi, 2, 0, "mat-icon", 3), t.j41(15, "span"), t.EFF(16), t.k0s()(), t.j41(17, "div", 1), t.nVh(18, Ei, 2, 0, "mat-icon", 2)(19, Mi, 2, 0, "mat-icon", 3), t.j41(20, "span"), t.EFF(21), t.k0s()(), t.j41(22, "div", 1), t.nVh(23, Ii, 2, 0, "mat-icon", 2)(24, Oi, 2, 0, "mat-icon", 3), t.j41(25, "span"), t.EFF(26), t.k0s()()()()), 2 & o && (t.R7$(3), t.vxM(i.passwordComponent.containAtLeastOneLowerCaseLetter ? 3 : 4), t.R7$(3), t.JRh(i.lowerCaseCriteriaMsg), t.R7$(2), t.vxM(i.passwordComponent.containAtLeastOneUpperCaseLetter ? 8 : 9), t.R7$(3), t.JRh(i.upperCaseCriteriaMsg), t.R7$(2), t.vxM(i.passwordComponent.containAtLeastOneDigit ? 13 : 14), t.R7$(3), t.JRh(i.digitsCriteriaMsg), t.R7$(2), t.vxM(i.passwordComponent.containAtLeastOneSpecialChar ? 18 : 19), t.R7$(3), t.JRh(i.specialCharsCriteriaMsg), t.R7$(2), t.vxM(i.passwordComponent.containAtLeastMinChars ? 23 : 24), t.R7$(3), t.JRh(i.minCharsCriteriaMsg))
                        },
                        dependencies: [h.Hu, h.RN, h.m2, f.m_, f.An],
                        styles: ["\n\n\n\n.info-card[_ngcontent-%COMP%]{animation:_ngcontent-%COMP%_cardEntrance .4s ease-out forwards;box-sizing:border-box;margin-top:1rem;width:100%}.info-card[_ngcontent-%COMP%]   mat-card-content[_ngcontent-%COMP%]{padding:16px}.info-row[_ngcontent-%COMP%]{align-items:center;animation:_ngcontent-%COMP%_rowEntrance .3s ease-out forwards;display:flex;gap:8px;margin:.5rem 0;opacity:0;transform:translate(-20px)}.info-row[_ngcontent-%COMP%]:nth-child(1){animation-delay:.1s}.info-row[_ngcontent-%COMP%]:nth-child(2){animation-delay:.2s}.info-row[_ngcontent-%COMP%]:nth-child(3){animation-delay:.3s}.info-row[_ngcontent-%COMP%]:nth-child(4){animation-delay:.4s}.info-row[_ngcontent-%COMP%]:nth-child(5){animation-delay:.5s}@keyframes _ngcontent-%COMP%_cardEntrance{0%{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:translateY(0)}}@keyframes _ngcontent-%COMP%_rowEntrance{to{opacity:1;transform:translate(0)}}"]
                    })
                }
                return n
            })();
            const fe = () => ({
                    length: "5-40"
                }),
                Pi = () => ({
                    value: 8
                });

            function ji(n, a) {
                1 & n && (t.j41(0, "mat-error", 7), t.EFF(1, "MANDATORY_EMAIL"), t.k0s())
            }

            function Ai(n, a) {
                1 & n && (t.j41(0, "mat-error", 7), t.EFF(1, "INVALID_EMAIL"), t.k0s())
            }

            function Di(n, a) {
                1 & n && (t.j41(0, "mat-error", 7), t.EFF(1, "MANDATORY_PASSWORD "), t.k0s())
            }

            function $i(n, a) {
                1 & n && (t.j41(0, "mat-error", 16), t.EFF(1, "INVALID_PASSWORD_LENGTH "), t.k0s()), 2 & n && t.Y8G("translateParams", t.lJ4(1, fe))
            }

            function Li(n, a) {
                1 & n && (t.j41(0, "mat-error", 7), t.EFF(1, " MANDATORY_PASSWORD_REPEAT "), t.k0s())
            }

            function Bi(n, a) {
                1 & n && (t.j41(0, "mat-error", 7), t.EFF(1, " PASSWORDS_NOT_MATCHING "), t.k0s())
            }

            function Ni(n, a) {
                if (1 & n && (t.nrm(0, "app-password-strength-info", 20), t.nI1(1, "translate"), t.nI1(2, "translate"), t.nI1(3, "translate"), t.nI1(4, "translate"), t.nI1(5, "translate")), 2 & n) {
                    t.XpG();
                    const e = t.sdS(42);
                    t.Y8G("passwordComponent", e)("lowerCaseCriteriaMsg", t.bMT(1, 6, "LOWER_CASE_CRITERIA_MSG"))("upperCaseCriteriaMsg", t.bMT(2, 8, "UPPER_CASE_CRITERIA_MSG"))("digitsCriteriaMsg", t.bMT(3, 10, "DIGITS_CRITERIA_MSG"))("specialCharsCriteriaMsg", t.bMT(4, 12, "SPECIAL_CHARS_CRITERIA_MSG"))("minCharsCriteriaMsg", t.i5U(5, 14, "MIN_CHARS_CRITERIA_MSG", t.lJ4(17, Pi)))
                }
            }

            function Ui(n, a) {
                if (1 & n && (t.j41(0, "mat-option", 24), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.Y8G("value", e.id), t.R7$(), t.SpI(" ", e.question, " ")
                }
            }

            function Vi(n, a) {
                1 & n && (t.j41(0, "mat-error", 7), t.EFF(1, " MANDATORY_SECURITY_QUESTION "), t.k0s())
            }

            function Gi(n, a) {
                1 & n && (t.j41(0, "mat-error", 7), t.EFF(1, " MANDATORY_SECURITY_ANSWER "), t.k0s())
            }
            S.Yv.add(C.nWR, C.tUE);
            let Yi = (() => {
                class n {
                    securityQuestionService;
                    userService;
                    securityAnswerService;
                    router;
                    formSubmitService;
                    translateService;
                    snackBar;
                    snackBarHelperService;
                    ngZone;
                    emailControl = new l.hs("", [l.k0.required, l.k0.email]);
                    passwordControl = new l.hs("", [l.k0.required, l.k0.minLength(5), l.k0.maxLength(40)]);
                    repeatPasswordControl = new l.hs("", [l.k0.required, Hi(this.passwordControl)]);
                    securityQuestionControl = new l.hs("", [l.k0.required]);
                    securityAnswerControl = new l.hs("", [l.k0.required]);
                    securityQuestions;
                    selected;
                    error = null;
                    constructor(e, o, i, r, c, p, k, x, O) {
                        this.securityQuestionService = e, this.userService = o, this.securityAnswerService = i, this.router = r, this.formSubmitService = c, this.translateService = p, this.snackBar = k, this.snackBarHelperService = x, this.ngZone = O
                    }
                    ngOnInit() {
                        this.securityQuestionService.find(null).subscribe({
                            next: e => {
                                this.securityQuestions = e
                            },
                            error: e => {
                                console.log(e)
                            }
                        }), this.formSubmitService.attachEnterKeyHandler("registration-form", "registerButton", () => {
                            this.save()
                        })
                    }
                    save() {
                        var e = this;
                        const o = {
                            email: this.emailControl.value,
                            password: this.passwordControl.value,
                            passwordRepeat: this.repeatPasswordControl.value,
                            securityQuestion: this.securityQuestions.find(i => i.id === this.securityQuestionControl.value),
                            securityAnswer: this.securityAnswerControl.value
                        };
                        this.userService.save(o).subscribe({
                            next: i => {
                                this.securityAnswerService.save({
                                    UserId: i.id,
                                    answer: this.securityAnswerControl.value,
                                    SecurityQuestionId: this.securityQuestionControl.value
                                }).subscribe(() => {
                                    this.ngZone.run((0, F.A)(function*() {
                                        return yield e.router.navigate(["/login"])
                                    })), this.snackBarHelperService.open("CONFIRM_REGISTER")
                                })
                            },
                            error: i => {
                                if (console.log(i), i.error?.errors) {
                                    const r = i.error.errors[0];
                                    this.error = r.message ? r.message[0].toUpperCase() + r.message.slice(1) : r
                                }
                            }
                        })
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(Gt), t.rXU(P), t.rXU(pe), t.rXU(R.Ix), t.rXU(tt), t.rXU(d.c$), t.rXU(rt.UG), t.rXU(L.r), t.rXU(t.SKi))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-register"]
                        ],
                        decls: 71,
                        vars: 38,
                        consts: [
                            ["password", ""],
                            ["repeatPassword", ""],
                            ["passwordInfoToggle", ""],
                            ["passwordStrength", ""],
                            [1, "center-container"],
                            ["appearance", "outlined", 1, "mat-elevation-z6"],
                            [1, "mdc-card"],
                            ["translate", ""],
                            ["ng-if", "error", 1, "error"],
                            ["id", "registration-form", 1, "form-container"],
                            ["appearance", "outline", "color", "accent"],
                            ["id", "emailControl", "type", "text", "matInput", "", "aria-label", "Email address field", 3, "focus", "formControl"],
                            ["id", "passwordControl", "type", "password", "matInput", "", "aria-label", "Field for the password", 3, "focus", "formControl"],
                            [1, "fas", "fa-exclamation-circle"],
                            ["translate", "", 2, "margin-left", "5px"],
                            ["align", "end"],
                            ["translate", "", 3, "translateParams"],
                            ["id", "repeatPasswordControl", "type", "password", "matInput", "", "aria-label", "Field to confirm the password", 3, "focus", "formControl"],
                            [3, "color"],
                            [3, "password"],
                            [3, "passwordComponent", "lowerCaseCriteriaMsg", "upperCaseCriteriaMsg", "digitsCriteriaMsg", "specialCharsCriteriaMsg", "minCharsCriteriaMsg"],
                            [1, "security-container"],
                            ["color", "accent", "appearance", "outline"],
                            ["placeholder", "", "name", "securityQuestion", "aria-label", "Selection list for the security question", 3, "valueChange", "focus", "formControl", "value"],
                            [1, "mat-body", 3, "value"],
                            ["id", "securityAnswerControl", "type", "text", "matInput", "", "aria-label", "Field for the answer to the security question", 3, "focus", "formControl", "placeholder"],
                            ["type", "submit", "id", "registerButton", "mat-raised-button", "", "color", "primary", "aria-label", "Button to complete the registration", 3, "click", "disabled"],
                            ["id", "alreadyACustomerLink"],
                            ["routerLink", "/login", "translate", "", 1, "primary-link"]
                        ],
                        template: function(o, i) {
                            if (1 & o) {
                                const r = t.RV6();
                                t.j41(0, "div", 4)(1, "mat-card", 5)(2, "div", 6)(3, "h1", 7), t.EFF(4, "TITLE_REGISTRATION"), t.k0s(), t.j41(5, "div", 8), t.EFF(6), t.k0s(), t.j41(7, "div", 9)(8, "mat-form-field", 10)(9, "mat-label", 7), t.EFF(10, "LABEL_EMAIL"), t.k0s(), t.j41(11, "input", 11), t.bIt("focus", function() {
                                    return s.eBV(r), s.Njj(i.error = null)
                                }), t.k0s(), t.nVh(12, ji, 2, 0, "mat-error", 7), t.nVh(13, Ai, 2, 0, "mat-error", 7), t.k0s(), t.j41(14, "mat-form-field", 10)(15, "mat-label", 7), t.EFF(16, "LABEL_PASSWORD"), t.k0s(), t.j41(17, "input", 12, 0), t.bIt("focus", function() {
                                    return s.eBV(r), s.Njj(i.error = null)
                                }), t.k0s(), t.j41(19, "mat-hint", 7), t.nrm(20, "i", 13), t.j41(21, "em", 14), t.EFF(22), t.nI1(23, "translate"), t.k0s()(), t.j41(24, "mat-hint", 15), t.EFF(25), t.k0s(), t.nVh(26, Di, 2, 0, "mat-error", 7), t.nVh(27, $i, 2, 2, "mat-error", 16), t.k0s(), t.j41(28, "mat-form-field", 10)(29, "mat-label", 7), t.EFF(30, "LABEL_PASSWORD_REPEAT"), t.k0s(), t.j41(31, "input", 17, 1), t.bIt("focus", function() {
                                    return s.eBV(r), s.Njj(i.error = null)
                                }), t.k0s(), t.j41(33, "mat-hint", 15), t.EFF(34), t.k0s(), t.nVh(35, Li, 2, 0, "mat-error", 7), t.nVh(36, Bi, 2, 0, "mat-error", 7), t.k0s(), t.j41(37, "mat-slide-toggle", 18, 2), t.EFF(39), t.nI1(40, "translate"), t.k0s(), t.nrm(41, "app-password-strength", 19, 3), t.nVh(43, Ni, 6, 18, "app-password-strength-info", 20), t.j41(44, "div", 21)(45, "mat-form-field", 22)(46, "mat-label"), t.EFF(47), t.nI1(48, "translate"), t.k0s(), t.j41(49, "mat-select", 23), t.mxI("valueChange", function(p) {
                                    return s.eBV(r), t.DH7(i.selected, p) || (i.selected = p), s.Njj(p)
                                }), t.bIt("focus", function() {
                                    return s.eBV(r), s.Njj(i.error = null)
                                }), t.Z7z(50, Ui, 2, 2, "mat-option", 24, t.fX1), t.k0s(), t.j41(52, "mat-hint", 7), t.nrm(53, "i", 13), t.j41(54, "em", 14), t.EFF(55, "CANNOT_BE_CHANGED_LATER"), t.k0s()(), t.nVh(56, Vi, 2, 0, "mat-error", 7), t.k0s(), t.j41(57, "mat-form-field", 10)(58, "mat-label", 7), t.EFF(59, "SECURITY_ANSWER"), t.k0s(), t.j41(60, "input", 25), t.nI1(61, "translate"), t.bIt("focus", function() {
                                    return s.eBV(r), s.Njj(i.error = null)
                                }), t.k0s(), t.nVh(62, Gi, 2, 0, "mat-error", 7), t.k0s()(), t.j41(63, "button", 26), t.bIt("click", function() {
                                    return s.eBV(r), s.Njj(i.save())
                                }), t.j41(64, "mat-icon"), t.EFF(65, "person_add"), t.k0s(), t.EFF(66), t.nI1(67, "translate"), t.k0s(), t.j41(68, "div", 27)(69, "a", 28), t.EFF(70, "ALREADY_A_CUSTOMER"), t.k0s()()()()()()
                            }
                            if (2 & o) {
                                const r = t.sdS(18),
                                    c = t.sdS(32),
                                    p = t.sdS(38),
                                    k = t.sdS(42);
                                t.R7$(6), t.JRh(i.error), t.R7$(5), t.Y8G("formControl", i.emailControl), t.R7$(), t.vxM(i.emailControl.invalid && i.emailControl.errors.required ? 12 : -1), t.R7$(), t.vxM(i.emailControl.invalid && i.emailControl.errors.email ? 13 : -1), t.R7$(4), t.Y8G("formControl", i.passwordControl), t.R7$(5), t.JRh(t.i5U(23, 26, "INVALID_PASSWORD_LENGTH", t.lJ4(37, fe))), t.R7$(3), t.SpI("", (null == r.value ? null : r.value.length) || 0, "/20"), t.R7$(), t.vxM(i.passwordControl.invalid && i.passwordControl.errors.required ? 26 : -1), t.R7$(), t.vxM(i.passwordControl.invalid && (i.passwordControl.errors.minlength || i.passwordControl.errors.maxlength) ? 27 : -1), t.R7$(4), t.Y8G("formControl", i.repeatPasswordControl), t.R7$(3), t.SpI("", (null == c.value ? null : c.value.length) || 0, "/40"), t.R7$(), t.vxM(i.repeatPasswordControl.invalid && i.repeatPasswordControl.errors.required ? 35 : -1), t.R7$(), t.vxM(i.repeatPasswordControl.invalid && i.repeatPasswordControl.errors.notSame ? 36 : -1), t.R7$(), t.Y8G("color", k.color), t.R7$(2), t.JRh(t.bMT(40, 29, "SHOW_PASSWORD_ADVICE")), t.R7$(2), t.Y8G("password", r.value), t.R7$(2), t.vxM(p.checked ? 43 : -1), t.R7$(4), t.SpI(" ", t.bMT(48, 31, "LABEL_SECURITY_QUESTION"), " "), t.R7$(2), t.Y8G("formControl", i.securityQuestionControl), t.R50("value", i.selected), t.R7$(), t.Dyx(i.securityQuestions), t.R7$(6), t.vxM(i.securityQuestionControl.invalid && i.securityQuestionControl.errors.required ? 56 : -1), t.R7$(4), t.Y8G("formControl", i.securityAnswerControl)("placeholder", t.bMT(61, 33, "SECURITY_ANSWER_PLACEHOLDER")), t.R7$(2), t.vxM(i.securityAnswerControl.invalid && i.securityAnswerControl.errors.required ? 62 : -1), t.R7$(), t.Y8G("disabled", i.emailControl.invalid || i.passwordControl.invalid || i.repeatPasswordControl.invalid || i.securityQuestionControl.invalid || i.securityAnswerControl.invalid), t.R7$(3), t.SpI(" ", t.bMT(67, 35, "BTN_REGISTER"), " ")
                            }
                        },
                        dependencies: [h.Hu, h.RN, d.h, d.Mm, I.R, _.rl, _.nJ, _.MV, _.TL, T.fS, T.fg, l.YN, l.me, l.BC, l.X1, l.l_, $t.sG, he, ge, St.VO, ue.wT, b.Hl, b.$z, nt.Wk, f.m_, f.An, d.D9],
                        styles: ["\n\n\n\nmat-card[_ngcontent-%COMP%]{height:auto;min-width:320px;width:25%}mat-form-field[_ngcontent-%COMP%]{padding-top:10px}.form-container[_ngcontent-%COMP%]{display:flex;flex-direction:column;position:relative}#registerButton[_ngcontent-%COMP%]{margin-left:20%;margin-top:30px;width:60%}#alreadyACustomerLink[_ngcontent-%COMP%]{align-self:center;font-size:12px;margin-top:40px}.security-container[_ngcontent-%COMP%]{display:flex;flex-direction:column;position:relative}  mat-option .mat-mdc-option-text{font-size:14px}.center-container[_ngcontent-%COMP%]{align-items:stretch;display:flex;justify-content:center}"]
                    })
                }
                return n
            })();

            function Hi(n) {
                return function(e) {
                    return n.value !== e.value ? {
                        notSame: !0
                    } : null
                }
            }
            const _e = () => ({
                    length: "5-40"
                }),
                Xi = () => ({
                    value: 8
                });

            function Wi(n, a) {
                1 & n && (t.j41(0, "mat-error", 7), t.EFF(1, "MANDATORY_EMAIL"), t.k0s())
            }

            function zi(n, a) {
                1 & n && (t.j41(0, "mat-error", 7), t.EFF(1, "INVALID_EMAIL"), t.k0s())
            }

            function Ji(n, a) {
                1 & n && (t.j41(0, "mat-error", 7), t.EFF(1, " MANDATORY_SECURITY_ANSWER "), t.k0s())
            }

            function Qi(n, a) {
                1 & n && (t.j41(0, "mat-error", 7), t.EFF(1, "MANDATORY_NEW_PASSWORD "), t.k0s())
            }

            function qi(n, a) {
                1 & n && (t.j41(0, "mat-error", 7), t.EFF(1, " MANDATORY_PASSWORD_REPEAT "), t.k0s())
            }

            function Ki(n, a) {
                1 & n && (t.j41(0, "mat-error", 7), t.EFF(1, " PASSWORDS_NOT_MATCHING "), t.k0s())
            }

            function Zi(n, a) {
                1 & n && (t.j41(0, "mat-error", 22), t.EFF(1, "INVALID_PASSWORD_LENGTH "), t.k0s()), 2 & n && t.Y8G("translateParams", t.lJ4(1, _e))
            }

            function ta(n, a) {
                if (1 & n && (t.nrm(0, "app-password-strength-info", 26), t.nI1(1, "translate"), t.nI1(2, "translate"), t.nI1(3, "translate"), t.nI1(4, "translate"), t.nI1(5, "translate")), 2 & n) {
                    t.XpG();
                    const e = t.sdS(56);
                    t.Y8G("passwordComponent", e)("lowerCaseCriteriaMsg", t.bMT(1, 6, "LOWER_CASE_CRITERIA_MSG"))("upperCaseCriteriaMsg", t.bMT(2, 8, "UPPER_CASE_CRITERIA_MSG"))("digitsCriteriaMsg", t.bMT(3, 10, "DIGITS_CRITERIA_MSG"))("specialCharsCriteriaMsg", t.bMT(4, 12, "SPECIAL_CHARS_CRITERIA_MSG"))("minCharsCriteriaMsg", t.i5U(5, 14, "MIN_CHARS_CRITERIA_MSG", t.lJ4(17, Xi)))
                }
            }
            S.Yv.add(C.hSh, H.MT);
            let ea = (() => {
                class n {
                    securityQuestionService;
                    userService;
                    translate;
                    emailControl = new l.hs("", [l.k0.required, l.k0.email]);
                    securityQuestionControl = new l.hs({
                        disabled: !0,
                        value: ""
                    }, [l.k0.required]);
                    passwordControl = new l.hs({
                        disabled: !0,
                        value: ""
                    }, [l.k0.required, l.k0.minLength(5)]);
                    repeatPasswordControl = new l.hs({
                        disabled: !0,
                        value: ""
                    }, [l.k0.required, na(this.passwordControl)]);
                    securityQuestion;
                    error;
                    confirmation;
                    timeoutDuration = 1e3;
                    timeout;
                    constructor(e, o, i) {
                        this.securityQuestionService = e, this.userService = o, this.translate = i
                    }
                    findSecurityQuestion() {
                        clearTimeout(this.timeout), this.timeout = setTimeout(() => {
                            this.securityQuestion = void 0, this.emailControl.value ? this.securityQuestionService.findBy(this.emailControl.value).subscribe({
                                next: e => {
                                    e ? (this.securityQuestion = e.question, this.securityQuestionControl.enable(), this.passwordControl.enable(), this.repeatPasswordControl.enable()) : (this.securityQuestionControl.disable(), this.passwordControl.disable(), this.repeatPasswordControl.disable())
                                },
                                error: e => e
                            }) : (this.securityQuestionControl.disable(), this.passwordControl.disable(), this.repeatPasswordControl.disable())
                        }, this.timeoutDuration)
                    }
                    resetPassword() {
                        this.userService.resetPassword({
                            email: this.emailControl.value,
                            answer: this.securityQuestionControl.value,
                            new: this.passwordControl.value,
                            repeat: this.repeatPasswordControl.value
                        }).subscribe({
                            next: () => {
                                this.error = void 0, this.translate.get("PASSWORD_SUCCESSFULLY_CHANGED").subscribe({
                                    next: e => {
                                        this.confirmation = e
                                    },
                                    error: e => {
                                        this.confirmation = e
                                    }
                                }), this.resetForm()
                            },
                            error: e => {
                                this.error = e.error, this.confirmation = void 0, this.resetErrorForm()
                            }
                        })
                    }
                    resetForm() {
                        this.emailControl.setValue(""), this.emailControl.markAsPristine(), this.emailControl.markAsUntouched(), this.securityQuestionControl.setValue(""), this.securityQuestionControl.markAsPristine(), this.securityQuestionControl.markAsUntouched(), this.passwordControl.setValue(""), this.passwordControl.markAsPristine(), this.passwordControl.markAsUntouched(), this.repeatPasswordControl.setValue(""), this.repeatPasswordControl.markAsPristine(), this.repeatPasswordControl.markAsUntouched()
                    }
                    resetErrorForm() {
                        this.emailControl.markAsPristine(), this.emailControl.markAsUntouched(), this.securityQuestionControl.setValue(""), this.securityQuestionControl.markAsPristine(), this.securityQuestionControl.markAsUntouched(), this.passwordControl.setValue(""), this.passwordControl.markAsPristine(), this.passwordControl.markAsUntouched(), this.repeatPasswordControl.setValue(""), this.repeatPasswordControl.markAsPristine(), this.repeatPasswordControl.markAsUntouched()
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(Gt), t.rXU(P), t.rXU(d.c$))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-forgot-password"]
                        ],
                        decls: 63,
                        vars: 42,
                        consts: [
                            ["password", ""],
                            ["repeatPassword", ""],
                            ["passwordInfoToggle", ""],
                            ["passwordStrength", ""],
                            [1, "center-container"],
                            ["appearance", "outlined", 1, "mat-elevation-z6"],
                            [1, "mdc-card"],
                            ["translate", ""],
                            [1, "confirmation", 3, "hidden"],
                            [1, "error", 3, "hidden"],
                            [1, "form-container"],
                            ["appearance", "outline", "color", "accent"],
                            ["id", "email", "type", "email", "matInput", "", "placeholder", "Enter your email", "aria-label", "Email address field", 3, "ngModelChange", "formControl"],
                            ["matSuffix", "", "matTooltipPosition", "right", "aria-label", "Please enter your email address to proceed", 3, "matTooltip"],
                            ["id", "forgot-form", 1, "form-container"],
                            ["id", "securityAnswer", "type", "password", "matInput", "", "aria-label", "Field for the answer to the security question", 3, "formControl", "placeholder"],
                            ["matSuffix", "", "matTooltipPosition", "right", "aria-label", "Please answer your selected security question", 3, "matTooltip"],
                            ["id", "newPassword", "type", "password", "matInput", "", "placeholder", "", "aria-label", "Field for New Password", 3, "formControl"],
                            [1, "fas", "fa-exclamation-circle"],
                            ["translate", "", 2, "margin-left", "5px"],
                            ["align", "end"],
                            ["id", "newPasswordRepeat", "type", "password", "matInput", "", "placeholder", "", "aria-label", "Field to confirm the new password", 3, "formControl"],
                            ["translate", "", 3, "translateParams"],
                            [3, "color"],
                            [3, "password"],
                            [1, "advice-container"],
                            [3, "passwordComponent", "lowerCaseCriteriaMsg", "upperCaseCriteriaMsg", "digitsCriteriaMsg", "specialCharsCriteriaMsg", "minCharsCriteriaMsg"],
                            ["type", "submit", "id", "resetButton", "mat-raised-button", "", "color", "primary", "aria-label", "Button to confirm the changes", 3, "click", "disabled"],
                            ["aria-hidden", "true", 1, "far", "fa-edit", "fa-lg"]
                        ],
                        template: function(o, i) {
                            if (1 & o) {
                                const r = t.RV6();
                                t.j41(0, "div", 4)(1, "mat-card", 5)(2, "div", 6)(3, "h1", 7), t.EFF(4, "TITLE_FORGOT_PASSWORD"), t.k0s(), t.j41(5, "div", 8), t.EFF(6), t.k0s(), t.j41(7, "div", 9), t.EFF(8), t.k0s(), t.j41(9, "div", 10)(10, "mat-form-field", 11)(11, "mat-label", 7), t.EFF(12, "LABEL_EMAIL"), t.k0s(), t.j41(13, "input", 12), t.bIt("ngModelChange", function() {
                                    return s.eBV(r), s.Njj(i.findSecurityQuestion())
                                }), t.k0s(), t.j41(14, "mat-icon", 13), t.nI1(15, "translate"), t.EFF(16, "help_outline "), t.k0s(), t.nVh(17, Wi, 2, 0, "mat-error", 7), t.nVh(18, zi, 2, 0, "mat-error", 7), t.k0s()(), t.j41(19, "div", 14)(20, "mat-form-field", 11)(21, "mat-label", 7), t.EFF(22, "LABEL_SECURITY_QUESTION"), t.k0s(), t.nrm(23, "input", 15), t.j41(24, "mat-icon", 16), t.nI1(25, "translate"), t.EFF(26, "help_outline "), t.k0s(), t.nVh(27, Ji, 2, 0, "mat-error", 7), t.k0s(), t.j41(28, "mat-form-field", 11)(29, "mat-label", 7), t.EFF(30, "LABEL_NEW_PASSWORD"), t.k0s(), t.nrm(31, "input", 17, 0), t.j41(33, "mat-hint", 7), t.nrm(34, "i", 18), t.j41(35, "em", 19), t.EFF(36), t.nI1(37, "translate"), t.k0s()(), t.j41(38, "mat-hint", 20), t.EFF(39), t.k0s(), t.nVh(40, Qi, 2, 0, "mat-error", 7), t.k0s(), t.j41(41, "mat-form-field", 11)(42, "mat-label", 7), t.EFF(43, "LABEL_REPEAT_NEW_PASSWORD"), t.k0s(), t.nrm(44, "input", 21, 1), t.j41(46, "mat-hint", 20), t.EFF(47), t.k0s(), t.nVh(48, qi, 2, 0, "mat-error", 7), t.nVh(49, Ki, 2, 0, "mat-error", 7), t.nVh(50, Zi, 2, 2, "mat-error", 22), t.k0s(), t.j41(51, "mat-slide-toggle", 23, 2), t.EFF(53), t.nI1(54, "translate"), t.k0s(), t.nrm(55, "app-password-strength", 24, 3), t.j41(57, "div", 25), t.nVh(58, ta, 6, 18, "app-password-strength-info", 26), t.k0s()(), t.j41(59, "button", 27), t.bIt("click", function() {
                                    return s.eBV(r), s.Njj(i.resetPassword())
                                }), t.nrm(60, "i", 28), t.EFF(61), t.nI1(62, "translate"), t.k0s()()()()
                            }
                            if (2 & o) {
                                const r = t.sdS(32),
                                    c = t.sdS(45),
                                    p = t.sdS(52),
                                    k = t.sdS(56);
                                t.R7$(5), t.Y8G("hidden", !(i.confirmation && !i.emailControl.dirty && !i.securityQuestionControl.dirty && !i.passwordControl.dirty && !i.repeatPasswordControl.dirty)), t.R7$(), t.SpI(" ", i.confirmation, " "), t.R7$(), t.Y8G("hidden", !(i.error && !i.emailControl.dirty && !i.securityQuestionControl.dirty && !i.passwordControl.dirty && !i.repeatPasswordControl.dirty)), t.R7$(), t.SpI(" ", i.error, " "), t.R7$(5), t.Y8G("formControl", i.emailControl), t.R7$(), t.Y8G("matTooltip", t.mNQ(t.bMT(15, 30, "MANDATORY_EMAIL"))), t.R7$(3), t.vxM(i.emailControl.invalid && i.emailControl.errors.required ? 17 : -1), t.R7$(), t.vxM(i.emailControl.invalid && i.emailControl.errors.email ? 18 : -1), t.R7$(5), t.Y8G("placeholder", t.mNQ(i.securityQuestion))("formControl", i.securityQuestionControl), t.R7$(), t.Y8G("matTooltip", t.mNQ(t.bMT(25, 32, "MANDATORY_SECURITY_ANSWER"))), t.R7$(3), t.vxM(i.securityQuestionControl.invalid && i.securityQuestionControl.errors.required ? 27 : -1), t.R7$(4), t.Y8G("formControl", i.passwordControl), t.R7$(5), t.JRh(t.i5U(37, 34, "INVALID_PASSWORD_LENGTH", t.lJ4(41, _e))), t.R7$(3), t.SpI("", (null == r.value ? null : r.value.length) || 0, "/20"), t.R7$(), t.vxM(i.passwordControl.invalid && i.passwordControl.errors.required ? 40 : -1), t.R7$(4), t.Y8G("formControl", i.repeatPasswordControl), t.R7$(3), t.SpI("", (null == c.value ? null : c.value.length) || 0, "/20"), t.R7$(), t.vxM(i.repeatPasswordControl.invalid && i.repeatPasswordControl.errors.required ? 48 : -1), t.R7$(), t.vxM(i.repeatPasswordControl.invalid && i.repeatPasswordControl.errors.notSame ? 49 : -1), t.R7$(), t.vxM(i.repeatPasswordControl.invalid && (null != i.repeatPasswordControl && i.repeatPasswordControl.errors.minlength || null != i.repeatPasswordControl && i.repeatPasswordControl.errors.maxlength) ? 50 : -1), t.R7$(), t.Y8G("color", k.color), t.R7$(2), t.JRh(t.bMT(54, 37, "SHOW_PASSWORD_ADVICE")), t.R7$(2), t.Y8G("password", r.value), t.R7$(3), t.vxM(p.checked ? 58 : -1), t.R7$(), t.Y8G("disabled", i.emailControl.invalid || i.securityQuestionControl.invalid || i.passwordControl.invalid || i.repeatPasswordControl.invalid || i.repeatPasswordControl.disabled), t.R7$(2), t.SpI(" ", t.bMT(62, 39, "BTN_CHANGE"), " ")
                            }
                        },
                        dependencies: [h.Hu, h.RN, d.h, d.Mm, I.R, _.rl, _.nJ, _.MV, _.TL, _.yw, T.fS, T.fg, l.YN, l.me, l.BC, l.X1, l.l_, f.m_, f.An, B.oV, $t.sG, he, ge, b.Hl, b.$z, d.D9],
                        styles: ['\n\n\n\nmat-card[_ngcontent-%COMP%]{height:auto;min-width:320px;width:25%}mat-form-field[_ngcontent-%COMP%]{padding-top:10px}.form-container[_ngcontent-%COMP%]{display:flex;flex-direction:column;position:relative}button[_ngcontent-%COMP%]{display:block;margin-left:auto;margin-right:auto;margin-top:30px;width:60%}mat-form-field[_ngcontent-%COMP%]   input[_ngcontent-%COMP%]{color:#fff!important}mat-form-field[_ngcontent-%COMP%]   .mat-input-element[_ngcontent-%COMP%]::placeholder{color:#ffffff80!important}#forgot-form[_ngcontent-%COMP%]   mat-hint[_ngcontent-%COMP%]{color:#b0b0b0;font-size:.8em;font-style:italic}.advice-container[_ngcontent-%COMP%]{display:flex;justify-content:center;margin-top:1rem;padding-top:15px;position:relative}.advice-container[_ngcontent-%COMP%]:before{background-color:#ff00004d;border-radius:2px;content:"";height:3px;left:0;position:absolute;right:0;top:0}.advice-container[_ngcontent-%COMP%]   app-password-strength-info[_ngcontent-%COMP%]{max-width:400px;width:100%}.center-container[_ngcontent-%COMP%]{align-items:stretch;display:flex;justify-content:center}']
                    })
                }
                return n
            })();

            function na(n) {
                return function(e) {
                    return n.value !== e.value ? {
                        notSame: !0
                    } : null
                }
            }
            let Yt = (() => {
                class n {
                    http;
                    hostServer = y.c.hostServer;
                    host = this.hostServer + "/rest/products";
                    constructor(e) {
                        this.http = e
                    }
                    get(e) {
                        return this.http.get(`${this.host}/${e}/reviews`).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                            throw o
                        }))
                    }
                    create(e, o) {
                        return this.http.put(`${this.host}/${e}/reviews`, o).pipe((0, v.T)(i => i.data), (0, g.W)(i => {
                            throw i
                        }))
                    }
                    patch(e) {
                        return this.http.patch(this.host + "/reviews", e).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                            throw o
                        }))
                    }
                    like(e) {
                        return this.http.post(this.host + "/reviews", {
                            id: e
                        }).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                            throw o
                        }))
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(s.KVO(E.Qq))
                    };
                    static \u0275prov = s.jDH({
                        token: n,
                        factory: n.\u0275fac,
                        providedIn: "root"
                    })
                }
                return n
            })();

            function oa(n, a) {
                1 & n && (t.j41(0, "mat-error", 2), t.EFF(1, "MANDATORY_REVIEW"), t.k0s())
            }
            S.Yv.add(C.isI, C._sz);
            let ia = (() => {
                class n {
                    data;
                    productReviewService;
                    dialogRef;
                    snackBarHelperService;
                    editReviewControl = new l.hs("", [l.k0.required, l.k0.minLength(1), l.k0.maxLength(160)]);
                    error = null;
                    constructor(e, o, i, r) {
                        this.data = e, this.productReviewService = o, this.dialogRef = i, this.snackBarHelperService = r
                    }
                    ngOnInit() {
                        this.editReviewControl.setValue(this.data.reviewData.message)
                    }
                    editReview() {
                        this.productReviewService.patch({
                            id: this.data.reviewData._id,
                            message: this.editReviewControl.value
                        }).subscribe({
                            next: () => {
                                this.dialogRef.close()
                            },
                            error: e => {
                                console.log(e), this.error = e
                            }
                        }), this.snackBarHelperService.open("CONFIRM_CHANGES_SAVED")
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(w.Vh), t.rXU(Yt), t.rXU(w.CP), t.rXU(L.r))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-product-review-edit"]
                        ],
                        decls: 27,
                        vars: 14,
                        consts: [
                            ["textPut", ""],
                            [1, "container", "column-layout"],
                            ["translate", ""],
                            ["appearance", "outline", "color", "accent", "hintLabel", "Max 160 characters", "floatLabel", "always"],
                            ["matInput", "", "matTextareaAutosize", "", "matAutosizeMinRows", "2", "maxlength", "160", "matAutosizeMaxRows", "4", "aria-label", "Text field to edit a product review", 3, "focus", "formControl", "placeholder"],
                            ["align", "end"],
                            ["mat-stroked-button", "", "mat-dialog-close", "", "aria-label", "Close Dialog", 1, "close-dialog"],
                            ["type", "submit", "mat-raised-button", "", "color", "primary", "aria-label", "Send the review", 2, "margin-left", "5px", 3, "click", "disabled"]
                        ],
                        template: function(o, i) {
                            if (1 & o) {
                                const r = t.RV6();
                                t.j41(0, "mat-dialog-content")(1, "div", 1)(2, "h1", 2), t.EFF(3, "LABEL_EDIT_REVIEW"), t.k0s(), t.j41(4, "div")(5, "mat-form-field", 3)(6, "mat-label", 2), t.EFF(7, "LABEL_REVIEW"), t.k0s(), t.j41(8, "textarea", 4, 0), t.nI1(10, "translate"), t.bIt("focus", function() {
                                    return s.eBV(r), s.Njj(i.error = null)
                                }), t.k0s(), t.j41(11, "mat-hint", 5), t.EFF(12), t.k0s(), t.nVh(13, oa, 2, 0, "mat-error", 2), t.k0s()()()(), t.j41(14, "mat-dialog-actions", 5)(15, "button", 6)(16, "mat-icon"), t.EFF(17, "close"), t.k0s(), t.j41(18, "span"), t.EFF(19), t.nI1(20, "translate"), t.k0s()(), t.j41(21, "button", 7), t.bIt("click", function() {
                                    return s.eBV(r), s.Njj(i.editReview())
                                }), t.j41(22, "mat-icon"), t.EFF(23, "send"), t.k0s(), t.j41(24, "span"), t.EFF(25), t.nI1(26, "translate"), t.k0s()()()
                            }
                            if (2 & o) {
                                const r = t.sdS(9);
                                t.R7$(8), t.Y8G("placeholder", t.mNQ(t.bMT(10, 8, "WRITE_REVIEW_PLACEHOLDER")))("formControl", i.editReviewControl), t.R7$(4), t.SpI("", (null == r.value ? null : r.value.length) || 0, "/160"), t.R7$(), t.vxM(i.editReviewControl.invalid ? 13 : -1), t.R7$(6), t.JRh(t.bMT(20, 10, "BTN_CLOSE")), t.R7$(2), t.Y8G("disabled", i.editReviewControl.invalid), t.R7$(4), t.JRh(t.bMT(26, 12, "BTN_SUBMIT"))
                            }
                        },
                        dependencies: [w.Yi, d.h, d.Mm, I.R, _.rl, _.nJ, _.MV, _.TL, T.fS, T.fg, l.YN, l.me, l.BC, l.tU, l.X1, l.l_, w.E7, b.Hl, b.$z, w.tx, f.m_, f.An, d.D9],
                        styles: ["\n\n\n\nmat-form-field[_ngcontent-%COMP%]{width:100%}.column-layout[_ngcontent-%COMP%]{display:flex;flex-direction:column}"]
                    })
                }
                return n
            })();
            const aa = () => ({
                length: "160"
            });

            function ra(n, a) {
                if (1 & n && (t.j41(0, "div", 7), t.nI1(1, "translate"), t.j41(2, "span", 21), t.nrm(3, "i", 22), t.j41(4, "span", 23), t.EFF(5), t.k0s()()()), 2 & n) {
                    const e = t.XpG();
                    t.Y8G("matTooltip", t.mNQ(t.bMT(1, 3, "LABEL_BONUS"))), t.R7$(5), t.JRh(e.data.productData.points)
                }
            }

            function sa(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "div", 24)(1, "div", 25)(2, "div", 26), t.nI1(3, "translate"), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG(2);
                        return s.Njj("Anonymous" !== i.author && i.author === r.author && r.editReview(i))
                    }), t.j41(4, "cite"), t.EFF(5), t.k0s(), t.j41(6, "p"), t.EFF(7), t.k0s()(), t.j41(8, "div", 27)(9, "button", 28), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG(2);
                        return s.Njj(r.likeReview(i))
                    }), t.j41(10, "mat-icon"), t.EFF(11, "thumb_up"), t.k0s(), t.j41(12, "span", 29), t.EFF(13), t.k0s()()()()()
                }
                if (2 & n) {
                    const e = a.$implicit,
                        o = t.XpG(2);
                    t.R7$(2), t.Y8G("matTooltipDisabled", t.mNQ(e.author !== o.author))("matTooltip", t.mNQ(t.bMT(3, 8, "LABEL_EDIT_REVIEW"))), t.R7$(3), t.JRh(e.author), t.R7$(2), t.JRh(e.message), t.R7$(2), t.Y8G("disabled", e.liked || !o.isLoggedIn()), t.R7$(4), t.JRh(e.likesCount)
                }
            }

            function la(n, a) {
                if (1 & n && (t.j41(0, "div"), t.Z7z(1, sa, 14, 10, "div", 24, t.fX1), t.nI1(3, "async"), t.k0s()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(), t.Dyx(t.bMT(3, 0, e.reviews$))
                }
            }

            function ca(n, a) {
                1 & n && (t.j41(0, "div")(1, "span", 30), t.EFF(2, " EMPTY_REVIEW_LIST "), t.k0s()())
            }
            S.Yv.add(C.isI, C._sz, C.XUe, C.Wcv, C.zqi);
            let Ce = (() => {
                    class n {
                        dialog;
                        data;
                        productReviewService;
                        userService;
                        snackBar;
                        snackBarHelperService;
                        author = "Anonymous";
                        reviews$;
                        userSubscription;
                        reviewControl = new l.hs("", [l.k0.maxLength(160)]);
                        constructor(e, o, i, r, c, p) {
                            this.dialog = e, this.data = o, this.productReviewService = i, this.userService = r, this.snackBar = c, this.snackBarHelperService = p
                        }
                        ngOnInit() {
                            this.data.productData.points = Math.round(this.data.productData.price / 10), this.reviews$ = this.productReviewService.get(this.data.productData.id), this.userSubscription = this.userService.whoAmI().subscribe({
                                next: e => {
                                    this.author = e?.email ? e.email : "Anonymous"
                                },
                                error: e => {
                                    console.log(e)
                                }
                            })
                        }
                        ngOnDestroy() {
                            this.userSubscription && this.userSubscription.unsubscribe()
                        }
                        addReview(e) {
                            const o = {
                                message: e.value,
                                author: this.author
                            };
                            e.value = "", this.productReviewService.create(this.data.productData.id, o).subscribe({
                                next: () => {
                                    this.reviews$ = this.productReviewService.get(this.data.productData.id)
                                },
                                error: i => {
                                    console.log(i)
                                }
                            }), this.snackBarHelperService.open("CONFIRM_REVIEW_SAVED")
                        }
                        editReview(e) {
                            this.dialog.open(ia, {
                                width: "500px",
                                height: "max-content",
                                data: {
                                    reviewData: e
                                }
                            }).afterClosed().subscribe(() => this.reviews$ = this.productReviewService.get(this.data.productData.id))
                        }
                        likeReview(e) {
                            this.productReviewService.like(e._id).subscribe(() => {
                                console.log("Liked " + e._id)
                            }), setTimeout(() => this.reviews$ = this.productReviewService.get(this.data.productData.id), 200)
                        }
                        isLoggedIn() {
                            return localStorage.getItem("token")
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(w.bZ), t.rXU(w.Vh), t.rXU(Yt), t.rXU(P), t.rXU(rt.UG), t.rXU(L.r))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-product-details"]
                            ],
                            decls: 58,
                            vars: 40,
                            consts: [
                                ["textPut", ""],
                                [1, "container"],
                                [1, "details-row"],
                                [1, "no-shrink"],
                                [1, "img-thumbnail", 3, "src", "alt"],
                                [3, "innerHTML"],
                                [1, "item-price"],
                                ["aria-label", "Bonus points when buying the product", 3, "matTooltip"],
                                [1, "detail-divider"],
                                ["mat-button", "", 2, "height", "0", "position", "absolute"],
                                ["aria-label", "Expand for Reviews", 1, "mat-elevation-z0"],
                                ["translate", "", 2, "margin-right", "5px"],
                                ["translate", ""],
                                ["appearance", "outline", "color", "accent", "floatLabel", "always"],
                                [1, "fas", "fa-exclamation-circle"],
                                [2, "margin-left", "5px"],
                                ["cols", "50", "matInput", "", "matTextareaAutosize", "", "matAutosizeMinRows", "2", "maxlength", "160", "matAutosizeMaxRows", "4", "aria-label", "Text field to review a product", 3, "formControl", "placeholder"],
                                ["align", "end"],
                                ["align", "end", 1, "dialogAction"],
                                ["mat-stroked-button", "", "mat-dialog-close", "", "aria-label", "Close Dialog", 1, "close-dialog", "buttons"],
                                ["type", "submit", "id", "submitButton", "mat-raised-button", "", "color", "primary", "aria-label", "Send the review", 1, "buttons", 2, "margin-left", "5px", 3, "click", "disabled"],
                                [1, "fa-2x", "fa-layers", "fa-fw"],
                                [1, "fas", "fa-crown"],
                                [1, "fa-layers-counter", "fa-layers-bottom-left", "fa-2x", "warn-notification", 2, "font-size", "47px"],
                                [1, "comment"],
                                [1, "review-row"],
                                ["matTooltipPosition", "right", 1, "review-text", 3, "click", "matTooltipDisabled", "matTooltip"],
                                [1, "like-container"],
                                ["mat-icon-button", "", "aria-label", "Rate a helpful review", 1, "rw-button", 3, "click", "disabled"],
                                [1, "like-counter", "accent-notification"],
                                ["translate", "", 1, "noResultText"]
                            ],
                            template: function(o, i) {
                                if (1 & o) {
                                    const r = t.RV6();
                                    t.j41(0, "mat-dialog-content")(1, "div", 1)(2, "div", 2)(3, "div", 3), t.nrm(4, "img", 4), t.k0s(), t.j41(5, "div")(6, "h1"), t.EFF(7), t.k0s(), t.nrm(8, "div", 5)(9, "br"), t.j41(10, "div")(11, "p", 6), t.EFF(12), t.k0s(), t.nVh(13, ra, 6, 5, "div", 7), t.k0s()()(), t.nrm(14, "mat-divider", 8)(15, "button", 9), t.j41(16, "mat-expansion-panel", 10)(17, "mat-expansion-panel-header")(18, "mat-panel-title")(19, "span", 11), t.EFF(20, "LABEL_REVIEWS"), t.k0s(), t.j41(21, "span"), t.EFF(22), t.nI1(23, "async"), t.k0s()()(), t.nrm(24, "button", 9), t.nVh(25, la, 4, 2, "div"), t.nI1(26, "async"), t.vZN(27, ca, 3, 0, "div"), t.k0s(), t.nrm(28, "mat-divider", 8), t.j41(29, "h4", 12), t.EFF(30, "WRITE_REVIEW"), t.k0s(), t.j41(31, "mat-form-field", 13)(32, "mat-label"), t.EFF(33), t.nI1(34, "translate"), t.k0s(), t.j41(35, "mat-hint"), t.nrm(36, "i", 14), t.j41(37, "em", 15), t.EFF(38), t.nI1(39, "translate"), t.k0s()(), t.nrm(40, "textarea", 16, 0), t.nI1(42, "translate"), t.j41(43, "mat-hint", 17), t.EFF(44), t.k0s()(), t.j41(45, "mat-dialog-actions", 18)(46, "button", 19)(47, "mat-icon"), t.EFF(48, "close"), t.k0s(), t.j41(49, "span"), t.EFF(50), t.nI1(51, "translate"), t.k0s()(), t.j41(52, "button", 20), t.bIt("click", function() {
                                        s.eBV(r);
                                        const p = t.sdS(41);
                                        return s.Njj(i.addReview(p))
                                    }), t.j41(53, "mat-icon"), t.EFF(54, "send"), t.k0s(), t.j41(55, "span"), t.EFF(56), t.nI1(57, "translate"), t.k0s()()()()()
                                }
                                if (2 & o) {
                                    let r, c;
                                    const p = t.sdS(41);
                                    t.R7$(4), t.Y8G("alt", t.mNQ(i.data.productData.name))("src", "assets/public/images/products/" + i.data.productData.image, t.B4B), t.R7$(3), t.JRh(i.data.productData.name), t.R7$(), t.Y8G("innerHTML", i.data.productData.description, t.npT), t.R7$(4), t.SpI("", i.data.productData.price, "\xa4"), t.R7$(), t.vxM(i.data.productData.points > 0 ? 13 : -1), t.R7$(9), t.SpI("(", null == (r = t.bMT(23, 24, i.reviews$)) ? null : r.length, ")"), t.R7$(3), t.vxM((null == (c = t.bMT(26, 26, i.reviews$)) ? null : c.length) >= 1 ? 25 : 27), t.R7$(4), t.xc7("display", i.isLoggedIn() ? "block" : "none"), t.R7$(2), t.xc7("display", i.isLoggedIn() ? "block" : "none"), t.R7$(2), t.JRh(t.bMT(34, 28, "LABEL_REVIEW")), t.R7$(5), t.JRh(t.i5U(39, 30, "MAX_TEXTAREA_LENGTH", t.lJ4(39, aa))), t.R7$(2), t.Y8G("placeholder", t.mNQ(t.bMT(42, 33, "WRITE_REVIEW_PLACEHOLDER")))("formControl", i.reviewControl), t.R7$(4), t.SpI("", (null == p.value ? null : p.value.length) || 0, "/160"), t.R7$(6), t.JRh(t.bMT(51, 35, "BTN_CLOSE")), t.R7$(2), t.xc7("display", i.isLoggedIn() ? "flex" : "none"), t.Y8G("disabled", !p.value.trim()), t.R7$(4), t.JRh(t.bMT(57, 37, "BTN_SUBMIT"))
                                }
                            },
                            dependencies: [w.Yi, B.oV, U.q, b.Hl, b.$z, V.iY, G.GK, G.Z2, G.WN, d.h, d.Mm, f.m_, f.An, I.R, _.rl, _.nJ, _.MV, T.fS, T.fg, l.YN, l.me, l.BC, l.tU, l.X1, l.l_, w.E7, w.tx, d.D9, Y.Jj],
                            styles: ["\n\n\n\n.img-thumbnail[_ngcontent-%COMP%]{height:auto;max-width:100%;padding:0;width:200px}mat-form-field[_ngcontent-%COMP%]{width:100%}p[_ngcontent-%COMP%]{word-break:break-all}.btn-container[_ngcontent-%COMP%]   button[_ngcontent-%COMP%]:nth-child(2){left:530px;width:50px}.detail-divider[_ngcontent-%COMP%]{margin-bottom:12px;margin-top:12px}.rw-button[_ngcontent-%COMP%]{margin-left:8px}.comment[_ngcontent-%COMP%]{margin-bottom:8px}.review-link[_ngcontent-%COMP%]{font-size:11px;margin-top:-10px;text-decoration:none}#submitButton[_ngcontent-%COMP%]{float:right}.item-price[_ngcontent-%COMP%]{float:left;margin-right:40px;margin-top:8px}.noResultText[_ngcontent-%COMP%]{display:block;font-size:small;margin-left:auto;margin-right:auto;text-align:center}.dialogAction[_ngcontent-%COMP%]{margin-bottom:0!important}.buttons[_ngcontent-%COMP%]{padding-left:13px;padding-right:13px}.like-button[_ngcontent-%COMP%]{align-items:center;display:flex;justify-content:center;margin-left:8px;margin-top:-4px;position:relative}.like-counter[_ngcontent-%COMP%]{border-radius:50%;bottom:14px;font-size:8px;padding:2px 4px;position:absolute;right:8px}h1[_ngcontent-%COMP%], div[_ngcontent-%COMP%]{color:var(--theme-text)}.mat-mdc-form-field-hint[_ngcontent-%COMP%]{align-items:center;color:var(--theme-text-fade-30);display:flex;font-size:11px;margin-left:-10px;margin-right:-10px}.container[_ngcontent-%COMP%]{display:flex;flex-direction:column}.details-row[_ngcontent-%COMP%]{display:flex;flex-direction:row;gap:20px}@media (max-width: 599.98px){.details-row[_ngcontent-%COMP%]{flex-direction:column}}.no-shrink[_ngcontent-%COMP%]{flex:0 0 auto}.review-row[_ngcontent-%COMP%]{align-items:flex-start;display:flex;gap:8px}.like-container[_ngcontent-%COMP%]{align-items:center;display:flex}"]
                        })
                    }
                    return n
                })(),
                Rt = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/api/Products";
                        constructor(e) {
                            this.http = e
                        }
                        search(e) {
                            return this.http.get(`${this.hostServer}/rest/products/search?q=${e}`).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        find(e) {
                            return this.http.get(this.host + "/", {
                                params: e
                            }).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        get(e) {
                            return this.http.get(`${this.host}/${e}?d=${encodeURIComponent((new Date).toDateString())}`).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        put(e, o) {
                            return this.http.put(`${this.host}/${e}`, o).pipe((0, v.T)(i => i.data), (0, g.W)(i => {
                                throw i
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })();
            var wt = m(7468);
            let Ht = (() => {
                class n {
                    http;
                    hostServer = y.c.hostServer;
                    host = this.hostServer + "/api/Quantitys";
                    constructor(e) {
                        this.http = e
                    }
                    getAll() {
                        return this.http.get(this.host + "/").pipe((0, v.T)(e => e.data), (0, g.W)(e => {
                            throw e
                        }))
                    }
                    put(e, o) {
                        return this.http.put(`${this.host}/${e}`, o).pipe((0, v.T)(i => i.data), (0, g.W)(i => {
                            throw i
                        }))
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(s.KVO(E.Qq))
                    };
                    static \u0275prov = s.jDH({
                        token: n,
                        factory: n.\u0275fac,
                        providedIn: "root"
                    })
                }
                return n
            })();
            const da = n => ({
                quantity: n
            });

            function ma(n, a) {
                if (1 & n && (t.j41(0, "div")(1, "span"), t.EFF(2), t.nI1(3, "translate"), t.k0s(), t.nrm(4, "span", 8), t.k0s()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(2), t.SpI("", t.bMT(3, 2, "TITLE_SEARCH_RESULTS"), " - "), t.R7$(2), t.Y8G("innerHTML", e.searchValue, t.npT)
                }
            }

            function pa(n, a) {
                1 & n && (t.j41(0, "div"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.JRh(t.bMT(2, 1, "TITLE_ALL_PRODUCTS")))
            }

            function ua(n, a) {
                if (1 & n && (t.j41(0, "div", 12)(1, "span", 21), t.EFF(2, "LABEL_ONLY_QUANTITY_LEFT"), t.k0s()()), 2 & n) {
                    const e = t.XpG().$implicit;
                    t.R7$(), t.Y8G("translateParams", t.eq3(1, da, e.quantity))
                }
            }

            function ha(n, a) {
                1 & n && (t.j41(0, "div", 13)(1, "span"), t.EFF(2), t.nI1(3, "translate"), t.k0s()()), 2 & n && (t.R7$(2), t.JRh(t.bMT(3, 1, "LABEL_SOLD_OUT")))
            }

            function ga(n, a) {
                if (1 & n && (t.j41(0, "span"), t.EFF(1), t.k0s()), 2 & n) {
                    const e = t.XpG().$implicit;
                    t.R7$(), t.SpI("", e.price, "\xa4")
                }
            }

            function fa(n, a) {
                if (1 & n && (t.j41(0, "span")(1, "s"), t.EFF(2), t.k0s(), t.EFF(3), t.k0s()), 2 & n) {
                    const e = t.XpG().$implicit;
                    t.R7$(2), t.JRh(e.price), t.R7$(), t.SpI(" ", e.deluxePrice, "\xa4 ")
                }
            }

            function _a(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 22), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG().$implicit,
                            r = t.XpG(2);
                        return s.Njj(r.addToBasket(i.id))
                    }), t.j41(1, "span"), t.EFF(2), t.nI1(3, "translate"), t.k0s()()
                }
                2 & n && (t.R7$(2), t.JRh(t.bMT(3, 1, "ADD_BASKET")))
            }

            function Ca(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-grid-tile")(1, "mat-card", 10)(2, "div", 11), t.nVh(3, ua, 3, 3, "div", 12), t.nVh(4, ha, 4, 3, "div", 13), t.j41(5, "div", 14), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG(2);
                        return s.Njj(r.showDetail(i))
                    }), t.nrm(6, "img", 15), t.j41(7, "div", 16)(8, "div", 17), t.EFF(9), t.k0s(), t.j41(10, "div", 18), t.nVh(11, ga, 2, 1, "span"), t.nVh(12, fa, 4, 2, "span"), t.k0s()()(), t.j41(13, "div", 19), t.nVh(14, _a, 4, 3, "button", 20), t.k0s()()()()
                }
                if (2 & n) {
                    const e = a.$implicit,
                        o = t.XpG(2);
                    t.R7$(), t.xc7("width", "100%"), t.R7$(2), t.vxM(e.quantity <= 5 && e.quantity > 0 ? 3 : -1), t.R7$(), t.vxM(e.quantity <= 0 ? 4 : -1), t.R7$(2), t.Y8G("alt", t.mNQ(e.name))("src", "assets/public/images/products/" + e.image, t.B4B), t.R7$(3), t.JRh(e.name), t.R7$(2), t.vxM(o.isDeluxe() && e.price !== e.deluxePrice ? -1 : 11), t.R7$(), t.vxM(o.isDeluxe() && e.price !== e.deluxePrice ? 12 : -1), t.R7$(2), t.vxM(o.isLoggedIn() ? 14 : -1)
                }
            }

            function ba(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "div")(1, "mat-grid-list", 9, 1), t.bIt("resize", function(i) {
                        s.eBV(e);
                        const r = t.XpG();
                        return s.Njj(r.onResize(i))
                    }, t.tSv), t.Z7z(3, Ca, 15, 11, "mat-grid-tile", null, t.fX1), t.nI1(5, "async"), t.k0s()()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.R7$(), t.Y8G("cols", e.breakpoint), t.R7$(2), t.Dyx(t.bMT(5, 1, e.gridDataSource))
                }
            }

            function va(n, a) {
                1 & n && (t.j41(0, "mat-card", 6)(1, "div", 11), t.nrm(2, "img", 23), t.j41(3, "mat-card-title")(4, "span", 24), t.EFF(5), t.nI1(6, "translate"), t.k0s()(), t.j41(7, "mat-card-content")(8, "span", 24), t.EFF(9), t.nI1(10, "translate"), t.k0s()()()()), 2 & n && (t.R7$(5), t.SpI(" ", t.bMT(6, 2, "NO_SEARCH_RESULT"), " "), t.R7$(4), t.SpI(" ", t.bMT(10, 4, "EMPTY_SEARCH_RESULT"), " "))
            }
            S.Yv.add(C.pS3, C.K$j);
            let Xt = (() => {
                    class n {
                        deluxeGuard;
                        dialog;
                        productService;
                        quantityService;
                        basketService;
                        translateService;
                        router;
                        route;
                        sanitizer;
                        ngZone;
                        io;
                        snackBarHelperService;
                        cdRef;
                        displayedColumns = ["Image", "Product", "Description", "Price", "Select"];
                        tableData;
                        pageSizeOptions = [];
                        dataSource;
                        gridDataSource;
                        searchValue;
                        resultsLength = 0;
                        paginator = null;
                        productSubscription;
                        routerSubscription;
                        breakpoint = 6;
                        emptyState = !1;
                        constructor(e, o, i, r, c, p, k, x, O, K, ht, gt, ft) {
                            this.deluxeGuard = e, this.dialog = o, this.productService = i, this.quantityService = r, this.basketService = c, this.translateService = p, this.router = k, this.route = x, this.sanitizer = O, this.ngZone = K, this.io = ht, this.snackBarHelperService = gt, this.cdRef = ft
                        }
                        ngAfterViewInit() {
                            const e = this.productService.search(""),
                                o = this.quantityService.getAll();
                            (0, wt.p)([o, e]).subscribe({
                                next: ([i, r]) => {
                                    const c = [];
                                    this.tableData = r, this.trustProductDescription(r);
                                    for (const k of r) c.push({
                                        name: k.name,
                                        price: k.price,
                                        deluxePrice: k.deluxePrice,
                                        id: k.id,
                                        image: k.image,
                                        description: k.description
                                    });
                                    for (const k of i) {
                                        const x = c.find(O => O.id === k.ProductId);
                                        void 0 !== x && (x.quantity = k.quantity)
                                    }
                                    this.dataSource = new u.I6(c);
                                    for (let k = 1; k <= Math.ceil(this.dataSource.data.length / 12); k++) this.pageSizeOptions.push(12 * k);
                                    this.paginator.pageSizeOptions = this.pageSizeOptions, this.dataSource.paginator = this.paginator, this.gridDataSource = this.dataSource.connect(), this.resultsLength = this.dataSource.data.length, this.filterTable(), this.routerSubscription = this.router.events.subscribe(() => {
                                        this.filterTable()
                                    });
                                    const p = this.route.snapshot.queryParams.challenge;
                                    p && this.route.snapshot.url.join("").match(/hacking-instructor/) && this.startHackingInstructor(decodeURIComponent(p)), window.innerWidth < 2600 ? (this.breakpoint = 4, window.innerWidth < 1740 && (this.breakpoint = 3, window.innerWidth < 1280 && (this.breakpoint = 2, window.innerWidth < 850 && (this.breakpoint = 1)))) : this.breakpoint = 6, this.cdRef.detectChanges()
                                },
                                error: i => {
                                    console.log(i)
                                }
                            })
                        }
                        trustProductDescription(e) {
                            for (let o = 0; o < e.length; o++) e[o].description = this.sanitizer.bypassSecurityTrustHtml(e[o].description)
                        }
                        ngOnDestroy() {
                            this.routerSubscription && this.routerSubscription.unsubscribe(), this.productSubscription && this.productSubscription.unsubscribe(), this.dataSource && this.dataSource.disconnect()
                        }
                        filterTable() {
                            let e = this.route.snapshot.queryParams.q;
                            e ? (e = e.trim(), this.ngZone.runOutsideAngular(() => {
                                this.io.socket().emit("verifyLocalXssChallenge", e)
                            }), this.dataSource.filter = e.toLowerCase(), this.searchValue = this.sanitizer.bypassSecurityTrustHtml(e), this.gridDataSource.subscribe(o => {
                                this.emptyState = 0 === o.length
                            })) : (this.dataSource.filter = "", this.searchValue = void 0, this.emptyState = !1)
                        }
                        startHackingInstructor(e) {
                            console.log(`Starting instructions for challenge "${e}"`), m.e(781).then(m.bind(m, 1143)).then(o => {
                                o.startHackingInstructorFor(e)
                            })
                        }
                        showDetail(e) {
                            this.dialog.open(Ce, {
                                width: "500px",
                                height: "max-content",
                                data: {
                                    productData: e
                                }
                            })
                        }
                        addToBasket(e) {
                            this.basketService.find(Number(sessionStorage.getItem("bid"))).subscribe({
                                next: o => {
                                    const i = o.Products;
                                    let r = !1;
                                    for (let c = 0; c < i.length; c++)
                                        if (i[c].id === e) {
                                            r = !0, this.basketService.get(i[c].BasketItem.id).subscribe({
                                                next: p => {
                                                    this.basketService.put(p.id, {
                                                        quantity: p.quantity + 1
                                                    }).subscribe({
                                                        next: x => {
                                                            this.productService.get(x.ProductId).subscribe({
                                                                next: O => {
                                                                    this.translateService.get("BASKET_ADD_SAME_PRODUCT", {
                                                                        product: O.name
                                                                    }).subscribe({
                                                                        next: K => {
                                                                            this.snackBarHelperService.open(K, "confirmBar"), this.basketService.updateNumberOfCartItems()
                                                                        },
                                                                        error: K => {
                                                                            this.snackBarHelperService.open(K, "confirmBar"), this.basketService.updateNumberOfCartItems()
                                                                        }
                                                                    })
                                                                },
                                                                error: O => {
                                                                    console.log(O)
                                                                }
                                                            })
                                                        },
                                                        error: x => {
                                                            this.snackBarHelperService.open(x.error?.error, "errorBar"), console.log(x)
                                                        }
                                                    })
                                                },
                                                error: p => {
                                                    console.log(p)
                                                }
                                            });
                                            break
                                        } r || this.basketService.save({
                                        ProductId: e,
                                        BasketId: sessionStorage.getItem("bid"),
                                        quantity: 1
                                    }).subscribe({
                                        next: c => {
                                            this.productService.get(c.ProductId).subscribe({
                                                next: p => {
                                                    this.translateService.get("BASKET_ADD_PRODUCT", {
                                                        product: p.name
                                                    }).subscribe({
                                                        next: k => {
                                                            this.snackBarHelperService.open(k, "confirmBar"), this.basketService.updateNumberOfCartItems()
                                                        },
                                                        error: k => {
                                                            this.snackBarHelperService.open(k, "confirmBar"), this.basketService.updateNumberOfCartItems()
                                                        }
                                                    })
                                                },
                                                error: p => {
                                                    console.log(p)
                                                }
                                            })
                                        },
                                        error: c => {
                                            this.snackBarHelperService.open(c.error?.error, "errorBar"), console.log(c)
                                        }
                                    })
                                },
                                error: o => {
                                    console.log(o)
                                }
                            })
                        }
                        isLoggedIn() {
                            return localStorage.getItem("token")
                        }
                        onResize(e) {
                            e.target.innerWidth < 2600 ? (this.breakpoint = 4, e.target.innerWidth < 1740 && (this.breakpoint = 3, e.target.innerWidth < 1280 && (this.breakpoint = 2, e.target.innerWidth < 850 && (this.breakpoint = 1)))) : this.breakpoint = 6
                        }
                        isDeluxe() {
                            return this.deluxeGuard.isDeluxe()
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(At), t.rXU(w.bZ), t.rXU(Rt), t.rXU(Ht), t.rXU(Q), t.rXU(d.c$), t.rXU(R.Ix), t.rXU(R.nX), t.rXU(it.up), t.rXU(t.SKi), t.rXU(et), t.rXU(L.r), t.rXU(D.gRc))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-search-result"]
                            ],
                            viewQuery: function(o, i) {
                                if (1 & o && t.GBs(mt.iy, 7), 2 & o) {
                                    let r;
                                    t.mGM(r = t.lsd()) && (i.paginator = r.first)
                                }
                            },
                            decls: 11,
                            vars: 5,
                            consts: [
                                ["paginator", ""],
                                ["table", ""],
                                [1, "center-align"],
                                [1, "table-container", "custom-slate"],
                                [1, "heading", "mat-elevation-z6"],
                                ["id", "search-result-heading"],
                                ["appearance", "outlined", 1, "mat-elevation-z6", "emptyState"],
                                ["color", "accent", 1, "mat-elevation-z6", 3, "pageSize", "pageSizeOptions", "length"],
                                ["id", "searchValue", 3, "innerHTML"],
                                ["gutterSize", "30px", 3, "resize", "cols"],
                                ["appearance", "outlined", 1, "mat-elevation-z6", "ribbon-card"],
                                [1, "mdc-card"],
                                [1, "ribbon", "ribbon-top-left"],
                                [1, "ribbon", "ribbon-top-left", "ribbon-sold"],
                                ["aria-label", "Click for more information about the product", "matTooltip", "Click for more information", "matTooltipPosition", "above", 1, "product", 3, "click"],
                                ["mat-card-image", "", "role", "button", 1, "img-responsive", "img-thumbnail", 3, "src", "alt"],
                                [1, "info-box"],
                                [1, "item-name"],
                                [1, "item-price"],
                                [1, "basket-btn-container", 2, "display", "flex", "justify-content", "center"],
                                ["aria-label", "Add to Basket", "color", "primary", "mat-button", "", "mat-raised-button", "", 1, "btn-basket"],
                                ["translate", "", 3, "translateParams"],
                                ["aria-label", "Add to Basket", "color", "primary", "mat-button", "", "mat-raised-button", "", 1, "btn-basket", 3, "click"],
                                ["alt", " No results found", "src", "assets/public/images/products/no-results.png", 1, "img-responsive", "noResult"],
                                [1, "noResultText"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "div", 2)(1, "div", 3)(2, "div", 4), t.nVh(3, ma, 5, 4, "div")(4, pa, 3, 3, "div"), t.nrm(5, "div", 5), t.k0s(), t.nVh(6, ba, 6, 3, "div")(7, va, 11, 6, "mat-card", 6), t.nrm(8, "mat-divider")(9, "mat-paginator", 7, 0), t.k0s()()), 2 & o && (t.R7$(3), t.vxM(i.searchValue ? 3 : 4), t.R7$(3), t.vxM(i.emptyState ? 7 : 6), t.R7$(3), t.Y8G("pageSize", 12)("pageSizeOptions", i.pageSizeOptions)("length", i.resultsLength))
                            },
                            dependencies: [Lt.B_, Lt.NS, h.Hu, h.RN, h.m2, h.kF, h.dh, d.h, d.Mm, B.oV, b.Hl, b.$z, U.q, mt.iy, d.D9, Y.Jj],
                            styles: ['\n\n\n\n.table-container[_ngcontent-%COMP%]{min-width:350px;width:80%}.heading[_ngcontent-%COMP%]{background:#0003;font-size:x-large;justify-content:center;padding:12px 20px}.img-thumbnail[_ngcontent-%COMP%]{align-self:start;width:60%!important}.heading[_ngcontent-%COMP%]{margin-bottom:10px}.product[_ngcontent-%COMP%]{align-items:stretch;cursor:pointer;display:flex;flex-direction:row;gap:12px}.item-price[_ngcontent-%COMP%]{font-size:initial;margin-top:12px;text-align:center}.item-name[_ngcontent-%COMP%]{font-size:large;text-align:center}.emptyState[_ngcontent-%COMP%]{display:block;height:auto!important;margin:20px auto;width:50%}.noResult[_ngcontent-%COMP%]{display:block;margin-left:auto;margin-right:auto;width:60%}.noResultText[_ngcontent-%COMP%]{display:block;margin-top:10px;text-align:center}.ribbon-card[_ngcontent-%COMP%]{position:relative}.ribbon[_ngcontent-%COMP%]{height:150px;overflow:hidden;pointer-events:none;position:absolute;width:150px}.ribbon[_ngcontent-%COMP%]:before{content:"";display:block;position:absolute;z-index:-1}.ribbon[_ngcontent-%COMP%]   span[_ngcontent-%COMP%]{box-shadow:0 5px 10px #0000001a;color:#fff;display:block;font:700 18px/1 Lato,sans-serif;padding:15px 0;position:absolute;text-align:center;text-shadow:0 1px 1px rgba(0,0,0,.2);width:225px}.ribbon-top-left[_ngcontent-%COMP%]{left:-10px;top:-10px;z-index:1}.ribbon-top-left[_ngcontent-%COMP%]:before{border-left-color:transparent;border-top-color:transparent;right:0;top:0}.ribbon-top-left[_ngcontent-%COMP%]   span[_ngcontent-%COMP%]{right:-25px;top:30px;transform:rotate(-45deg)}.img-responsive[_ngcontent-%COMP%]{height:auto;object-fit:cover;width:100%}.btn-basket[_ngcontent-%COMP%]{margin-top:20px}.mdc-card[_ngcontent-%COMP%]{border:0;display:flex;flex-direction:column;height:calc(100% - 20px);justify-content:space-between;overflow:hidden}mat-card[_ngcontent-%COMP%]{height:calc(100% - 70px);width:100%}mat-grid-tile[_ngcontent-%COMP%]{align-items:stretch;height:auto!important}.mat-divider.mat-divider-horizontal[_ngcontent-%COMP%]{margin-top:10px}.product[_ngcontent-%COMP%]{margin-left:-17px;margin-right:6px;margin-top:-17px}.center-align[_ngcontent-%COMP%]{display:flex;justify-content:center}.info-box[_ngcontent-%COMP%]{align-self:center;width:40%}@media (max-width: 959.98px){.product[_ngcontent-%COMP%]{flex-direction:column;gap:10px}.info-box[_ngcontent-%COMP%]{width:100%}.img-thumbnail[_ngcontent-%COMP%]{width:100%!important}}']
                        })
                    }
                    return n
                })(),
                be = (() => {
                    class n {
                        get nativeWindow() {
                            return function ka() {
                                return window
                            }()
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })();

            function Sa(n, a) {
                if (1 & n && (t.j41(0, "div", 5), t.EFF(1), t.k0s()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(), t.JRh(e.error)
                }
            }

            function Fa(n, a) {
                1 & n && (t.j41(0, "mat-error", 8), t.EFF(1, "MANDATORY_EMAIL"), t.k0s())
            }

            function xa(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 20), t.nI1(1, "translate"), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.hide = !i.hide)
                    }), t.nrm(2, "i", 21), t.k0s()
                }
                2 & n && t.Y8G("matTooltip", t.mNQ(t.bMT(1, 2, "SHOW_PWD_TOOLTIP")))
            }

            function Ta(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 22), t.nI1(1, "translate"), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.hide = !i.hide)
                    }), t.nrm(2, "i", 23), t.k0s()
                }
                2 & n && t.Y8G("matTooltip", t.mNQ(t.bMT(1, 2, "HIDE_PWD_TOOLTIP")))
            }

            function ya(n, a) {
                1 & n && (t.j41(0, "mat-error", 8), t.EFF(1, "MANDATORY_PASSWORD"), t.k0s())
            }

            function Ra(n, a) {
                1 & n && (t.j41(0, "div", 16)(1, "div", 24), t.nrm(2, "div"), t.k0s(), t.j41(3, "div", 25), t.EFF(4, "LABEL_OR"), t.k0s(), t.j41(5, "div", 24), t.nrm(6, "div"), t.k0s()())
            }

            function wa(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 26), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.googleLogin())
                    }), t.nrm(1, "i", 27), t.EFF(2), t.nI1(3, "translate"), t.k0s()
                }
                2 & n && (t.R7$(2), t.SpI(" ", t.bMT(3, 1, "BTN_GOOGLE_LOGIN"), " "))
            }
            S.Yv.add(C.bMg, C.pS3, C.k6j, N.FjH);
            let Ma = (() => {
                class n {
                    configurationService;
                    userService;
                    windowRefService;
                    cookieService;
                    router;
                    formSubmitService;
                    basketService;
                    ngZone;
                    emailControl = new l.hs("", [l.k0.required]);
                    passwordControl = new l.hs("", [l.k0.required, l.k0.minLength(1)]);
                    hide = !0;
                    user;
                    rememberMe = new l.hs(!1);
                    error;
                    clientId = "1005568560502-6hm16lef8oh46hr2d98vf2ohlnj4nfhq.apps.googleusercontent.com";
                    oauthUnavailable = !0;
                    redirectUri = "";
                    testingUsername = "testing@juice-sh.op";
                    testingPassword = "IamUsedForTesting";
                    constructor(e, o, i, r, c, p, k, x) {
                        this.configurationService = e, this.userService = o, this.windowRefService = i, this.cookieService = r, this.router = c, this.formSubmitService = p, this.basketService = k, this.ngZone = x
                    }
                    ngOnInit() {
                        const e = localStorage.getItem("email");
                        e ? (this.user = {}, this.user.email = e, this.rememberMe.setValue(!0)) : this.rememberMe.setValue(!1), this.redirectUri = `${this.windowRefService.nativeWindow.location.protocol}//${this.windowRefService.nativeWindow.location.host}`, this.configurationService.getApplicationConfiguration().subscribe({
                            next: o => {
                                if (o?.application?.googleOauth) {
                                    this.clientId = o.application.googleOauth.clientId;
                                    const i = o.application.googleOauth.authorizedRedirects.find(r => r.uri === this.redirectUri);
                                    i ? (this.oauthUnavailable = !1, this.redirectUri = i.proxy ? i.proxy : i.uri) : (this.oauthUnavailable = !0, console.log(this.redirectUri + " is not an authorized redirect URI for this application."))
                                }
                            },
                            error: o => {
                                console.log(o)
                            }
                        }), this.formSubmitService.attachEnterKeyHandler("login-form", "loginButton", () => {
                            this.login()
                        })
                    }
                    login() {
                        var e = this;
                        this.user = {}, this.user.email = this.emailControl.value, this.user.password = this.passwordControl.value, this.userService.login(this.user).subscribe({
                            next: o => {
                                localStorage.setItem("token", o.token);
                                const i = new Date;
                                i.setHours(i.getHours() + 8), this.cookieService.put("token", o.token, {
                                    expires: i
                                }), sessionStorage.setItem("bid", o.bid), this.basketService.updateNumberOfCartItems(), this.userService.isLoggedIn.next(!0), this.ngZone.run((0, F.A)(function*() {
                                    return yield e.router.navigate(["/search"])
                                }))
                            },
                            error: ({
                                error: o
                            }) => {
                                if (o.status && o.data && "totp_token_required" === o.status) return localStorage.setItem("totp_tmp_token", o.data.tmpToken), void this.ngZone.run((0, F.A)(function*() {
                                    return yield e.router.navigate(["/2fa/enter"])
                                }));
                                localStorage.removeItem("token"), this.cookieService.remove("token"), sessionStorage.removeItem("bid"), this.error = o, this.userService.isLoggedIn.next(!1), this.emailControl.markAsPristine(), this.passwordControl.markAsPristine()
                            }
                        }), this.rememberMe.value ? localStorage.setItem("email", this.user.email) : localStorage.removeItem("email")
                    }
                    googleLogin() {
                        this.windowRefService.nativeWindow.location.replace(`https://accounts.google.com/o/oauth2/v2/auth?client_id=${this.clientId}&response_type=token&scope=email&redirect_uri=${this.redirectUri}`)
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(M.s), t.rXU(P), t.rXU(be), t.rXU($.O1), t.rXU(R.Ix), t.rXU(tt), t.rXU(Q), t.rXU(t.SKi))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-login"]
                        ],
                        decls: 36,
                        vars: 17,
                        consts: [
                            ["email", ""],
                            ["password", ""],
                            [1, "center-container"],
                            ["appearance", "outlined", 1, "mat-elevation-z6"],
                            [1, "mdc-card"],
                            [1, "error"],
                            ["id", "login-form", 1, "form-container"],
                            ["color", "accent", "appearance", "outline"],
                            ["translate", ""],
                            ["id", "email", "name", "email", "matInput", "", "placeholder", "", "aria-label", "Text field for the login email", 3, "focus", "formControl"],
                            ["id", "password", "name", "password", "matInput", "", "placeholder", "", "aria-label", "Text field for the login password", 3, "focus", "formControl", "type"],
                            ["mat-icon-button", "", "matSuffix", "", "aria-label", "Button to display the password", "matTooltipPosition", "right", 3, "matTooltip"],
                            ["mat-icon-button", "", "matSuffix", "", "aria-label", "Button to hide the password", "matTooltipPosition", "right", 3, "matTooltip"],
                            ["routerLink", "/forgot-password", "translate", "", 1, "primary-link", "forgot-pw"],
                            ["type", "submit", "id", "loginButton", "mat-raised-button", "", "color", "primary", "aria-label", "Login", 3, "click", "disabled"],
                            ["id", "rememberMe", "aria-label", "Checkbox to stay logged in or not logged in", 3, "formControl"],
                            [1, "breakLine"],
                            ["id", "loginButtonGoogle", "mat-raised-button", "", "color", "accent", "aria-label", "Login with Google", 1, "google-button"],
                            ["id", "newCustomerLink"],
                            ["routerLink", "/register", "translate", "", 1, "primary-link"],
                            ["mat-icon-button", "", "matSuffix", "", "aria-label", "Button to display the password", "matTooltipPosition", "right", 3, "click", "matTooltip"],
                            ["aria-label", "Eye", 1, "fas", "fa-eye"],
                            ["mat-icon-button", "", "matSuffix", "", "aria-label", "Button to hide the password", "matTooltipPosition", "right", 3, "click", "matTooltip"],
                            ["aria-label", "Eye Slash", 1, "fas", "fa-eye-slash"],
                            [1, "line"],
                            ["translate", "", 1, "textOnLine"],
                            ["id", "loginButtonGoogle", "mat-raised-button", "", "color", "accent", "aria-label", "Login with Google", 1, "google-button", 3, "click"],
                            [1, "fab", "fa-google", "fa-lg"]
                        ],
                        template: function(o, i) {
                            if (1 & o) {
                                const r = t.RV6();
                                t.j41(0, "div", 2)(1, "mat-card", 3)(2, "div", 4)(3, "h1"), t.EFF(4, "Login"), t.k0s(), t.nVh(5, Sa, 2, 1, "div", 5), t.j41(6, "div", 6)(7, "mat-form-field", 7)(8, "mat-label", 8), t.EFF(9, "LABEL_EMAIL"), t.k0s(), t.j41(10, "input", 9, 0), t.bIt("focus", function() {
                                    return s.eBV(r), s.Njj(i.error = null)
                                }), t.k0s(), t.nVh(12, Fa, 2, 0, "mat-error", 8), t.k0s(), t.j41(13, "mat-form-field", 7)(14, "mat-label", 8), t.EFF(15, "LABEL_PASSWORD"), t.k0s(), t.j41(16, "input", 10, 1), t.bIt("focus", function() {
                                    return s.eBV(r), s.Njj(i.error = null)
                                }), t.k0s(), t.nVh(18, xa, 3, 4, "button", 11)(19, Ta, 3, 4, "button", 12), t.nVh(20, ya, 2, 0, "mat-error", 8), t.k0s(), t.j41(21, "a", 13), t.EFF(22, "FORGOT_PASSWORD"), t.k0s(), t.j41(23, "button", 14), t.bIt("click", function() {
                                    return s.eBV(r), s.Njj(i.login())
                                }), t.j41(24, "mat-icon"), t.EFF(25, " exit_to_app "), t.k0s(), t.EFF(26), t.nI1(27, "translate"), t.k0s(), t.j41(28, "mat-checkbox", 15), t.EFF(29), t.nI1(30, "translate"), t.k0s(), t.nVh(31, Ra, 7, 0, "div", 16), t.nVh(32, wa, 4, 3, "button", 17), t.j41(33, "div", 18)(34, "a", 19), t.EFF(35, "NO_CUSTOMER"), t.k0s()()()()()()
                            }
                            2 & o && (t.R7$(5), t.vxM(i.error ? 5 : -1), t.R7$(5), t.Y8G("formControl", i.emailControl), t.R7$(2), t.vxM(i.emailControl.invalid ? 12 : -1), t.R7$(4), t.Y8G("formControl", i.passwordControl)("type", i.hide ? "password" : "text"), t.R7$(2), t.vxM(i.hide ? 18 : 19), t.R7$(2), t.vxM(i.passwordControl.invalid ? 20 : -1), t.R7$(3), t.Y8G("disabled", !i.emailControl.value || !i.passwordControl.value), t.R7$(3), t.SpI(" ", t.bMT(27, 13, "BTN_LOGIN"), " "), t.R7$(2), t.Y8G("formControl", i.rememberMe), t.R7$(), t.SpI(" ", t.bMT(30, 15, "REMEMBER_ME"), " "), t.R7$(2), t.vxM(i.oauthUnavailable ? -1 : 31), t.R7$(), t.vxM(i.oauthUnavailable ? -1 : 32))
                        },
                        dependencies: [h.Hu, h.RN, I.R, _.rl, _.nJ, _.TL, _.yw, d.h, d.Mm, T.fS, T.fg, l.YN, l.me, l.BC, l.X1, l.l_, V.iY, B.oV, nt.Wk, b.Hl, b.$z, f.m_, f.An, Nt.So, d.D9],
                        styles: ["\n\n\n\nmat-card[_ngcontent-%COMP%]{height:auto;min-width:320px;width:25%}.form-container[_ngcontent-%COMP%]{display:flex;flex-direction:column;position:relative}.forgot-pw[_ngcontent-%COMP%]{font-size:11px;margin-top:-15px}mat-checkbox[_ngcontent-%COMP%]{margin-bottom:20px;margin-left:20%;margin-top:5px}mat-form-field[_ngcontent-%COMP%]{padding-top:10px}mat-form-field[_ngcontent-%COMP%]:nth-child(2){padding-bottom:20px}#loginButton[_ngcontent-%COMP%], #loginButtonGoogle[_ngcontent-%COMP%]{margin-left:20%;margin-top:30px;width:60%}.breakLine[_ngcontent-%COMP%]{display:table}.breakLine[_ngcontent-%COMP%] > div[_ngcontent-%COMP%]{display:table-cell;white-space:nowrap}.line[_ngcontent-%COMP%] > div[_ngcontent-%COMP%]{border-bottom:1px solid #c0c0c0;height:1px}.textOnLine[_ngcontent-%COMP%]{padding:0 5px}.line[_ngcontent-%COMP%]{vertical-align:middle;width:50%}#newCustomerLink[_ngcontent-%COMP%]{align-self:center;font-size:12px;margin-top:40px}.center-container[_ngcontent-%COMP%]{align-items:stretch;display:flex;justify-content:center}"]
                    })
                }
                return n
            })();
            S.Yv.add(C._sz);
            let Ia = (() => {
                    class n {
                        dialogData;
                        userService;
                        user;
                        constructor(e, o) {
                            this.dialogData = e, this.userService = o
                        }
                        ngOnInit() {
                            this.userService.get(this.dialogData.id).subscribe({
                                next: e => {
                                    this.user = e
                                },
                                error: e => {
                                    console.log(e)
                                }
                            })
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(w.Vh), t.rXU(P))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-user-details"]
                            ],
                            decls: 29,
                            vars: 10,
                            consts: [
                                [1, "container"],
                                [1, "row"],
                                ["translate", "LABEL_EMAIL"],
                                ["translate", "LABEL_CREATED_AT"],
                                ["translate", "LABEL_UPDATED_AT"],
                                ["align", "end", 1, "dialogAction"],
                                ["mat-button", "", 2, "height", "0", "position", "absolute"],
                                ["mat-stroked-button", "", "mat-dialog-close", "", "aria-label", "Close Dialog", 1, "close-dialog", "buttons"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "mat-dialog-content")(1, "header")(2, "h1"), t.EFF(3), t.nI1(4, "translate"), t.k0s()(), t.nrm(5, "mat-divider"), t.j41(6, "div", 0)(7, "div", 1)(8, "div"), t.nrm(9, "strong", 2), t.j41(10, "p"), t.EFF(11), t.k0s()()(), t.j41(12, "div", 1)(13, "div"), t.nrm(14, "strong", 3), t.j41(15, "p"), t.EFF(16), t.k0s()(), t.j41(17, "div"), t.nrm(18, "strong", 4), t.j41(19, "p"), t.EFF(20), t.k0s()()()(), t.j41(21, "mat-dialog-actions", 5), t.nrm(22, "button", 6), t.j41(23, "button", 7)(24, "mat-icon"), t.EFF(25, "close"), t.k0s(), t.j41(26, "span"), t.EFF(27), t.nI1(28, "translate"), t.k0s()()()()), 2 & o && (t.R7$(3), t.Lme("", t.bMT(4, 6, "LABEL_USER"), " #", null == i.user ? null : i.user.id), t.R7$(8), t.JRh(null == i.user ? null : i.user.email), t.R7$(5), t.JRh(null == i.user ? null : i.user.createdAt), t.R7$(4), t.JRh(null == i.user ? null : i.user.updatedAt), t.R7$(7), t.JRh(t.bMT(28, 8, "BTN_CLOSE")))
                            },
                            dependencies: [w.Yi, U.q, d.h, d.Mm, w.E7, b.Hl, b.$z, w.tx, f.m_, f.An, d.D9],
                            styles: ["\n\n\n\nmat-divider[_ngcontent-%COMP%]{margin-bottom:10px}.dialogAction[_ngcontent-%COMP%]{margin-bottom:0!important}.container[_ngcontent-%COMP%]{display:flex;flex-direction:column;gap:10px}.row[_ngcontent-%COMP%]{align-items:baseline;display:flex;gap:10px}"]
                        })
                    }
                    return n
                })(),
                Oa = (() => {
                    class n {
                        dialogData;
                        feedback;
                        id;
                        constructor(e) {
                            this.dialogData = e
                        }
                        ngOnInit() {
                            this.feedback = this.dialogData.feedback, this.id = this.dialogData.id
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(w.Vh))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-feedback-details"]
                            ],
                            decls: 23,
                            vars: 11,
                            consts: [
                                [1, "container"],
                                [1, "user-line"],
                                [2, "margin-top", "10px !important"],
                                [3, "innerHTML"],
                                ["align", "end", 1, "dialogAction"],
                                ["mat-button", "", 2, "height", "0", "position", "absolute"],
                                ["mat-stroked-button", "", "mat-dialog-close", "", "aria-label", "Close Dialog", 1, "close-dialog", "buttons"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "mat-dialog-content")(1, "header")(2, "h1"), t.EFF(3), t.nI1(4, "translate"), t.k0s()(), t.nrm(5, "mat-divider"), t.j41(6, "div", 0)(7, "div", 1)(8, "span"), t.EFF(9), t.nI1(10, "translate"), t.k0s(), t.j41(11, "p"), t.EFF(12), t.k0s()(), t.j41(13, "div", 2), t.nrm(14, "cite", 3), t.k0s()(), t.j41(15, "mat-dialog-actions", 4), t.nrm(16, "button", 5), t.j41(17, "button", 6)(18, "mat-icon"), t.EFF(19, "close"), t.k0s(), t.j41(20, "span"), t.EFF(21), t.nI1(22, "translate"), t.k0s()()()()), 2 & o && (t.R7$(3), t.JRh(t.bMT(4, 5, "LABEL_COMMENT")), t.R7$(6), t.SpI("", t.bMT(10, 7, "LABEL_USER"), ":"), t.R7$(3), t.SpI("#", i.id), t.R7$(2), t.Y8G("innerHTML", i.feedback, t.npT), t.R7$(7), t.JRh(t.bMT(22, 9, "BTN_CLOSE")))
                            },
                            dependencies: [w.Yi, U.q, w.E7, b.Hl, b.$z, w.tx, d.h, f.m_, f.An, d.D9],
                            styles: ["\n\n\n\nmat-divider[_ngcontent-%COMP%]{margin-bottom:10px}.dialogAction[_ngcontent-%COMP%]{margin-bottom:0!important}mat-dialog-content[_ngcontent-%COMP%]{max-width:320px;min-width:280px}.container[_ngcontent-%COMP%]{display:flex;flex-direction:column}.user-line[_ngcontent-%COMP%]{align-items:baseline;display:flex;gap:10px}"]
                        })
                    }
                    return n
                })();
            const Pa = ["paginatorUsers"],
                ja = ["paginatorFeedb"];

            function Aa(n, a) {
                1 & n && t.nrm(0, "mat-header-cell")
            }

            function Da(n, a) {
                1 & n && t.nrm(0, "i", 28)
            }

            function $a(n, a) {
                if (1 & n && (t.j41(0, "mat-cell", 27), t.nVh(1, Da, 1, 0, "i", 28), t.k0s()), 2 & n) {
                    const e = a.$implicit,
                        o = t.XpG();
                    t.R7$(), t.vxM(o.doesUserHaveAnActiveSession(e) ? 1 : -1)
                }
            }

            function La(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 4)
            }

            function Ba(n, a) {
                1 & n && t.nrm(0, "mat-cell", 29), 2 & n && t.Y8G("innerHTML", a.$implicit.email, t.npT)
            }

            function Na(n, a) {
                1 & n && t.nrm(0, "mat-header-cell")
            }

            function Ua(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-cell")(1, "button", 30), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG();
                        return s.Njj(r.showUserDetail(i.id))
                    }), t.nrm(2, "i", 31), t.k0s()()
                }
            }

            function Va(n, a) {
                1 & n && t.nrm(0, "mat-header-row")
            }

            function Ga(n, a) {
                1 & n && t.nrm(0, "mat-row")
            }

            function Ya(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 4)
            }

            function Ha(n, a) {
                if (1 & n && (t.j41(0, "mat-cell"), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.SpI(" ", e.UserId)
                }
            }

            function Xa(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 4)
            }

            function Wa(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-cell", 32), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG();
                        return s.Njj(r.showFeedbackDetails(i.comment, i.UserId))
                    }), t.nrm(1, "p", 33), t.k0s()
                }
                if (2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.Y8G("innerHTML", e.comment, t.npT)
                }
            }

            function za(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 4)
            }

            function Ja(n, a) {
                1 & n && (t.j41(0, "mat-icon"), t.EFF(1, " star_rate "), t.k0s())
            }

            function Qa(n, a) {
                if (1 & n && (t.j41(0, "mat-cell"), t.Z7z(1, Ja, 2, 0, "mat-icon", null, t.fX1), t.k0s()), 2 & n) {
                    const e = a.$implicit,
                        o = t.XpG();
                    t.R7$(), t.Dyx(o.times(e.rating))
                }
            }

            function qa(n, a) {
                1 & n && t.nrm(0, "mat-header-cell")
            }

            function Ka(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-cell")(1, "button", 30), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG();
                        return s.Njj(r.deleteFeedback(i.id))
                    }), t.nrm(2, "i", 34), t.k0s()()
                }
            }

            function Za(n, a) {
                1 & n && t.nrm(0, "mat-header-row")
            }

            function tr(n, a) {
                1 & n && t.nrm(0, "mat-row")
            }

            function er(n, a) {
                1 & n && t.nrm(0, "mat-header-cell")
            }

            function nr(n, a) {
                1 & n && t.nrm(0, "mat-cell", 27)
            }

            function or(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 4)
            }

            function ir(n, a) {
                1 & n && t.nrm(0, "mat-cell", 29), 2 & n && t.Y8G("innerHTML", a.$implicit.email, t.npT)
            }

            function ar(n, a) {
                1 & n && t.nrm(0, "mat-header-cell")
            }

            function rr(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-cell")(1, "button", 30), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG();
                        return s.Njj(r.showUserDetail(i.id))
                    }), t.nrm(2, "i", 31), t.k0s()()
                }
            }

            function sr(n, a) {
                1 & n && t.nrm(0, "mat-header-row")
            }

            function lr(n, a) {
                1 & n && t.nrm(0, "mat-row")
            }
            S.Yv.add(C.X46, C.pS3, C.v02, C.Oh6, C.BeE);
            let cr = (() => {
                class n {
                    dialog;
                    userService;
                    feedbackService;
                    sanitizer;
                    userDataSource;
                    userDataSourceHidden;
                    userColumns = ["user", "email", "user_detail"];
                    feedbackDataSource;
                    feedbackColumns = ["user", "comment", "rating", "remove"];
                    error;
                    resultsLengthUser = 0;
                    resultsLengthFeedback = 0;
                    paginatorUsers;
                    paginatorFeedb;
                    constructor(e, o, i, r) {
                        this.dialog = e, this.userService = o, this.feedbackService = i, this.sanitizer = r
                    }
                    ngOnInit() {
                        this.findAllUsers(), this.findAllFeedbacks()
                    }
                    findAllUsers() {
                        this.userService.find().subscribe({
                            next: e => {
                                this.userDataSource = e, this.userDataSourceHidden = e;
                                for (const o of this.userDataSource) o.email = this.sanitizer.bypassSecurityTrustHtml(`<span class="${this.doesUserHaveAnActiveSession(o)?"confirmation":"error"}">${o.email}</span>`);
                                this.userDataSource = new u.I6(this.userDataSource), this.userDataSource.paginator = this.paginatorUsers, this.resultsLengthUser = e.length
                            },
                            error: e => {
                                this.error = e, console.log(this.error)
                            }
                        })
                    }
                    findAllFeedbacks() {
                        this.feedbackService.find().subscribe({
                            next: e => {
                                this.feedbackDataSource = e;
                                for (const o of this.feedbackDataSource) o.comment = this.sanitizer.bypassSecurityTrustHtml(o.comment);
                                this.feedbackDataSource = new u.I6(this.feedbackDataSource), this.feedbackDataSource.paginator = this.paginatorFeedb, this.resultsLengthFeedback = e.length
                            },
                            error: e => {
                                this.error = e, console.log(this.error)
                            }
                        })
                    }
                    deleteFeedback(e) {
                        this.feedbackService.del(e).subscribe({
                            next: () => {
                                this.findAllFeedbacks()
                            },
                            error: o => {
                                this.error = o, console.log(this.error)
                            }
                        })
                    }
                    showUserDetail(e) {
                        this.dialog.open(Ia, {
                            data: {
                                id: e
                            }
                        })
                    }
                    showFeedbackDetails(e, o) {
                        this.dialog.open(Oa, {
                            data: {
                                feedback: e,
                                id: o
                            }
                        })
                    }
                    times(e) {
                        return Array(e).fill("\u2605")
                    }
                    doesUserHaveAnActiveSession(e) {
                        return e.lastLoginTime && e.lastLoginTime > Date.now() / 1e3 - 21600
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(w.bZ), t.rXU(P), t.rXU(Tt), t.rXU(it.up))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-administration"]
                        ],
                        viewQuery: function(o, i) {
                            if (1 & o && (t.GBs(Pa, 5), t.GBs(ja, 5)), 2 & o) {
                                let r;
                                t.mGM(r = t.lsd()) && (i.paginatorUsers = r.first), t.mGM(r = t.lsd()) && (i.paginatorFeedb = r.first)
                            }
                        },
                        decls: 57,
                        vars: 13,
                        consts: [
                            ["paginatorUsers", ""],
                            ["paginatorFeedb", ""],
                            ["appearance", "outlined", 1, "mat-elevation-z6", "mat-own-card"],
                            [1, "mdc-card"],
                            ["translate", ""],
                            [1, "container"],
                            [1, "user-table"],
                            [1, "heading"],
                            [3, "dataSource"],
                            ["matColumnDef", "user"],
                            [4, "matHeaderCellDef"],
                            ["style", "vertical-align: middle", 4, "matCellDef"],
                            ["matColumnDef", "email"],
                            ["translate", "", 4, "matHeaderCellDef"],
                            ["style", "vertical-align: middle", 3, "innerHTML", 4, "matCellDef"],
                            ["matColumnDef", "user_detail"],
                            [4, "matCellDef"],
                            [4, "matHeaderRowDef"],
                            [4, "matRowDef", "matRowDefColumns"],
                            ["color", "accent", 1, "mat-elevation-z0", 3, "pageSize", "length"],
                            [1, "customer-table"],
                            ["matColumnDef", "comment"],
                            [3, "click", 4, "matCellDef"],
                            ["matColumnDef", "rating"],
                            ["matColumnDef", "remove"],
                            [2, "visibility", "hidden", "height", "0px", 3, "dataSource"],
                            ["src", "assets/public/images/padding/19px.png"],
                            [2, "vertical-align", "middle"],
                            [1, "fas", "fa-user", "fa-lg", "confirmation"],
                            [2, "vertical-align", "middle", 3, "innerHTML"],
                            ["mat-button", "", 3, "click"],
                            [1, "fas", "fa-eye"],
                            [3, "click"],
                            ["matTooltip", "Click for more information", "matTooltipPosition", "above", 3, "innerHTML"],
                            [1, "fas", "fa-trash-alt"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "mat-card", 2)(1, "div", 3)(2, "h1", 4), t.EFF(3, "TITLE_ADMINISTRATION"), t.k0s(), t.j41(4, "div", 5)(5, "div", 6)(6, "div", 7)(7, "span", 4), t.EFF(8, "SECTION_USER"), t.k0s()(), t.j41(9, "mat-table", 8), t.qex(10, 9), t.DNE(11, Aa, 1, 0, "mat-header-cell", 10)(12, $a, 2, 1, "mat-cell", 11), t.bVm(), t.qex(13, 12), t.DNE(14, La, 1, 0, "mat-header-cell", 13)(15, Ba, 1, 1, "mat-cell", 14), t.bVm(), t.qex(16, 15), t.DNE(17, Na, 1, 0, "mat-header-cell", 10)(18, Ua, 3, 0, "mat-cell", 16), t.bVm(), t.DNE(19, Va, 1, 0, "mat-header-row", 17)(20, Ga, 1, 0, "mat-row", 18), t.k0s(), t.nrm(21, "mat-paginator", 19, 0), t.k0s(), t.j41(23, "div", 20)(24, "div", 7)(25, "span", 4), t.EFF(26, "SECTION_CUSTOMER_FEEDBACK"), t.k0s()(), t.j41(27, "mat-table", 8), t.qex(28, 9), t.DNE(29, Ya, 1, 0, "mat-header-cell", 13)(30, Ha, 2, 1, "mat-cell", 16), t.bVm(), t.qex(31, 21), t.DNE(32, Xa, 1, 0, "mat-header-cell", 13)(33, Wa, 2, 1, "mat-cell", 22), t.bVm(), t.qex(34, 23), t.DNE(35, za, 1, 0, "mat-header-cell", 13)(36, Qa, 3, 0, "mat-cell", 16), t.bVm(), t.qex(37, 24), t.DNE(38, qa, 1, 0, "mat-header-cell", 10)(39, Ka, 3, 0, "mat-cell", 16), t.bVm(), t.DNE(40, Za, 1, 0, "mat-header-row", 17)(41, tr, 1, 0, "mat-row", 18), t.k0s(), t.nrm(42, "mat-paginator", 19, 1), t.k0s()()()(), t.j41(44, "mat-table", 25), t.qex(45, 9), t.DNE(46, er, 1, 0, "mat-header-cell", 10)(47, nr, 1, 0, "mat-cell", 11), t.bVm(), t.qex(48, 12), t.DNE(49, or, 1, 0, "mat-header-cell", 13)(50, ir, 1, 1, "mat-cell", 14), t.bVm(), t.qex(51, 15), t.DNE(52, ar, 1, 0, "mat-header-cell", 10)(53, rr, 3, 0, "mat-cell", 16), t.bVm(), t.DNE(54, sr, 1, 0, "mat-header-row", 17)(55, lr, 1, 0, "mat-row", 18), t.k0s(), t.nrm(56, "img", 26)), 2 & o && (t.R7$(9), t.Y8G("dataSource", i.userDataSource), t.R7$(10), t.Y8G("matHeaderRowDef", i.userColumns), t.R7$(), t.Y8G("matRowDefColumns", i.userColumns), t.R7$(), t.Y8G("pageSize", 10)("length", i.resultsLengthUser), t.R7$(6), t.Y8G("dataSource", i.feedbackDataSource), t.R7$(13), t.Y8G("matHeaderRowDef", i.feedbackColumns), t.R7$(), t.Y8G("matRowDefColumns", i.feedbackColumns), t.R7$(), t.Y8G("pageSize", 10)("length", i.resultsLengthFeedback), t.R7$(2), t.Y8G("dataSource", i.userDataSourceHidden), t.R7$(10), t.Y8G("matHeaderRowDef", i.userColumns), t.R7$(), t.Y8G("matRowDefColumns", i.userColumns))
                        },
                        dependencies: [h.Hu, h.RN, d.h, d.Mm, u.Zl, u.cC, u.tL, u.KS, u.YV, u.$R, b.Hl, b.$z, u.ji, u.YZ, u.iL, u.NB, mt.iy, B.oV, f.m_, f.An],
                        styles: ["\n\n\n\n.heading[_ngcontent-%COMP%]{background:#00000021;font-size:large;justify-content:center;margin-bottom:10px;padding:12px}.container[_ngcontent-%COMP%]{display:grid;gap:48px;grid-template-columns:1fr 1fr}@media (max-width: 1100px){.container[_ngcontent-%COMP%]{gap:24px;grid-template-columns:1fr}}mat-table[_ngcontent-%COMP%]{height:800px}mat-header-row[_ngcontent-%COMP%]{min-height:0!important}.mat-column-icon[_ngcontent-%COMP%]   svg[_ngcontent-%COMP%], .mat-column-user_detail[_ngcontent-%COMP%]   button[_ngcontent-%COMP%]{margin-left:38%}mat-cell[_ngcontent-%COMP%], mat-footer-cell[_ngcontent-%COMP%], mat-header-cell[_ngcontent-%COMP%]{align-items:normal;display:block}.mat-column-comment[_ngcontent-%COMP%]{flex:0 0 55%!important;hyphens:auto;overflow-wrap:break-word;padding:5px;white-space:unset!important;width:55%!important;word-break:break-word;word-wrap:break-word!important}.mat-column-user[_ngcontent-%COMP%]{flex:0 0 7%!important;hyphens:auto;overflow-wrap:break-word;padding:5px;white-space:unset!important;width:7%!important;word-break:break-word;word-wrap:break-word!important}.mat-column-rating[_ngcontent-%COMP%]{flex:0 0 20%!important;hyphens:auto;overflow-wrap:break-word;white-space:unset!important;width:20%!important;word-break:break-word;word-wrap:break-word!important}mat-cell[_ngcontent-%COMP%]:last-child, mat-footer-cell[_ngcontent-%COMP%]:last-child, mat-header-cell[_ngcontent-%COMP%]:last-child{padding-right:50px}mat-row[_ngcontent-%COMP%]{height:69px}.mat-own-card[_ngcontent-%COMP%]{display:block;margin-bottom:20px;margin-left:10%;margin-right:10%}p[_ngcontent-%COMP%]{-webkit-box-orient:vertical;display:-webkit-box;line-clamp:2;overflow:hidden}"]
                    })
                }
                return n
            })();
            const ve = () => ({
                length: "5-40"
            });

            function dr(n, a) {
                1 & n && (t.j41(0, "mat-error"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.JRh(t.bMT(2, 1, "MANDATORY_CURRENT_PASSWORD")))
            }

            function mr(n, a) {
                1 & n && (t.j41(0, "mat-error"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.SpI(" ", t.bMT(2, 1, "MANDATORY_NEW_PASSWORD"), " "))
            }

            function pr(n, a) {
                1 & n && (t.j41(0, "mat-error", 15), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.Y8G("translateParams", t.lJ4(4, ve)), t.R7$(), t.SpI("", t.bMT(2, 2, "INVALID_PASSWORD_LENGTH"), " "))
            }

            function ur(n, a) {
                1 & n && (t.j41(0, "mat-error", 11), t.EFF(1, "MANDATORY_PASSWORD_REPEAT"), t.k0s())
            }

            function hr(n, a) {
                1 & n && (t.j41(0, "mat-error", 11), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.SpI(" ", t.bMT(2, 1, "PASSWORDS_NOT_MATCHING"), " "))
            }
            S.Yv.add(C.hSh, H.MT);
            let gr = (() => {
                class n {
                    userService;
                    formSubmitService;
                    translate;
                    passwordControl = new l.hs("", [l.k0.required]);
                    newPasswordControl = new l.hs("", [l.k0.required, l.k0.minLength(5), l.k0.maxLength(40)]);
                    repeatNewPasswordControl = new l.hs("", [l.k0.required, l.k0.minLength(5), l.k0.maxLength(40), fr(this.newPasswordControl)]);
                    error;
                    confirmation;
                    constructor(e, o, i) {
                        this.userService = e, this.formSubmitService = o, this.translate = i
                    }
                    ngOnInit() {
                        this.formSubmitService.attachEnterKeyHandler("password-form", "changeButton", () => {
                            this.changePassword()
                        })
                    }
                    changePassword() {
                        localStorage.getItem("email")?.match(/support@.*/) && !this.newPasswordControl.value.match(/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{12,30}/) && console.error("Parola echipei de asisten\u021b\u0103 nu respect\u0103 politica corporativ\u0103 pentru conturile privilegiate! V\u0103 rug\u0103m s\u0103 schimba\u021bi parola \xeen consecin\u021b\u0103!"), this.userService.changePassword({
                            current: this.passwordControl.value,
                            new: this.newPasswordControl.value,
                            repeat: this.repeatNewPasswordControl.value
                        }).subscribe({
                            next: () => {
                                this.error = void 0, this.translate.get("PASSWORD_SUCCESSFULLY_CHANGED").subscribe({
                                    next: e => {
                                        this.confirmation = e
                                    },
                                    error: e => {
                                        this.confirmation = {
                                            error: e
                                        }
                                    }
                                }), this.resetForm()
                            },
                            error: e => {
                                console.log(e), this.error = e, this.confirmation = void 0, this.resetPasswords()
                            }
                        })
                    }
                    resetForm() {
                        this.passwordControl.setValue(""), this.resetPasswords()
                    }
                    resetPasswords() {
                        this.passwordControl.markAsPristine(), this.passwordControl.markAsUntouched(), this.newPasswordControl.setValue(""), this.newPasswordControl.markAsPristine(), this.newPasswordControl.markAsUntouched(), this.repeatNewPasswordControl.setValue(""), this.repeatNewPasswordControl.markAsPristine(), this.repeatNewPasswordControl.markAsUntouched()
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(P), t.rXU(tt), t.rXU(d.c$))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-change-password"]
                        ],
                        decls: 47,
                        vars: 39,
                        consts: [
                            ["password", ""],
                            ["passwordRepeat", ""],
                            [1, "center-container"],
                            ["appearance", "outlined", 1, "mat-elevation-z6", 2, "margin-bottom", "20px"],
                            [1, "mdc-card"],
                            [1, "confirmation", 3, "hidden"],
                            [1, "error", 3, "hidden"],
                            ["id", "password-form", 1, "form-container"],
                            ["appearance", "outline", "color", "accent"],
                            ["id", "currentPassword", "type", "password", "matInput", "", "aria-label", "Field to enter the current password", 3, "formControl", "placeholder"],
                            ["id", "newPassword", "type", "password", "matInput", "", "aria-label", "Field for the new password", 3, "formControl"],
                            ["translate", ""],
                            [1, "fas", "fa-exclamation-circle"],
                            ["translate", "", 2, "margin-left", "5px"],
                            ["align", "end"],
                            ["translate", "", 3, "translateParams"],
                            ["id", "newPasswordRepeat", "type", "password", "matInput", "", "aria-label", "Field to repeat the new password", 3, "formControl"],
                            ["type", "submit", "id", "changeButton", "mat-raised-button", "", "color", "primary", "aria-label", "Button to confirm the change", 3, "click", "disabled"],
                            ["aria-hidden", "true", 1, "far", "fa-edit", "fa-lg"]
                        ],
                        template: function(o, i) {
                            if (1 & o) {
                                const r = t.RV6();
                                t.j41(0, "div", 2)(1, "mat-card", 3)(2, "div", 4)(3, "h1"), t.EFF(4), t.nI1(5, "translate"), t.k0s(), t.j41(6, "div", 5), t.EFF(7), t.k0s(), t.j41(8, "div", 6), t.EFF(9), t.k0s(), t.j41(10, "div", 7)(11, "mat-form-field", 8)(12, "mat-label"), t.EFF(13), t.nI1(14, "translate"), t.k0s(), t.nrm(15, "input", 9), t.nI1(16, "translate"), t.nVh(17, dr, 3, 3, "mat-error"), t.k0s(), t.j41(18, "mat-form-field", 8)(19, "mat-label"), t.EFF(20), t.nI1(21, "translate"), t.k0s(), t.nrm(22, "input", 10, 0), t.j41(24, "mat-hint", 11), t.nrm(25, "i", 12), t.j41(26, "em", 13), t.EFF(27), t.nI1(28, "translate"), t.k0s()(), t.j41(29, "mat-hint", 14), t.EFF(30), t.k0s(), t.nVh(31, mr, 3, 3, "mat-error"), t.nVh(32, pr, 3, 5, "mat-error", 15), t.k0s(), t.j41(33, "mat-form-field", 8)(34, "mat-label"), t.EFF(35), t.nI1(36, "translate"), t.k0s(), t.nrm(37, "input", 16, 1), t.j41(39, "mat-hint", 14), t.EFF(40), t.k0s(), t.nVh(41, ur, 2, 0, "mat-error", 11), t.nVh(42, hr, 3, 3, "mat-error", 11), t.k0s()(), t.j41(43, "button", 17), t.bIt("click", function() {
                                    return s.eBV(r), s.Njj(i.changePassword())
                                }), t.nrm(44, "i", 18), t.EFF(45), t.nI1(46, "translate"), t.k0s()()()()
                            }
                            if (2 & o) {
                                const r = t.sdS(23),
                                    c = t.sdS(38);
                                t.R7$(4), t.JRh(t.bMT(5, 23, "TITLE_CHANGE_PASSWORD")), t.R7$(2), t.Y8G("hidden", !(i.confirmation && !i.passwordControl.dirty && !i.newPasswordControl.dirty && !i.repeatNewPasswordControl.dirty)), t.R7$(), t.SpI(" ", i.confirmation, " "), t.R7$(), t.Y8G("hidden", !(i.error && !i.passwordControl.dirty && !i.newPasswordControl.dirty && !i.repeatNewPasswordControl.dirty)), t.R7$(), t.SpI(" ", i.error, " "), t.R7$(4), t.JRh(t.bMT(14, 25, "LABEL_CURRENT_PASSWORD")), t.R7$(2), t.Y8G("placeholder", t.mNQ(t.bMT(16, 27, "MANDATORY_CURRENT_PASSWORD")))("formControl", i.passwordControl), t.R7$(2), t.vxM(i.passwordControl.invalid ? 17 : -1), t.R7$(3), t.JRh(t.bMT(21, 29, "LABEL_NEW_PASSWORD")), t.R7$(2), t.Y8G("formControl", i.newPasswordControl), t.R7$(5), t.JRh(t.i5U(28, 31, "INVALID_PASSWORD_LENGTH", t.lJ4(38, ve))), t.R7$(3), t.SpI("", (null == r.value ? null : r.value.length) || 0, "/40"), t.R7$(), t.vxM(null != i.newPasswordControl && i.newPasswordControl.invalid && null != i.newPasswordControl && i.newPasswordControl.errors.required ? 31 : -1), t.R7$(), t.vxM(null != i.newPasswordControl && i.newPasswordControl.invalid && (null != i.newPasswordControl && i.newPasswordControl.errors.minlength || null != i.newPasswordControl && i.newPasswordControl.errors.maxlength) ? 32 : -1), t.R7$(3), t.JRh(t.bMT(36, 34, "LABEL_REPEAT_NEW_PASSWORD")), t.R7$(2), t.Y8G("formControl", i.repeatNewPasswordControl), t.R7$(3), t.SpI("", (null == c.value ? null : c.value.length) || 0, "/20"), t.R7$(), t.vxM(i.repeatNewPasswordControl.invalid && i.repeatNewPasswordControl.errors.required ? 41 : -1), t.R7$(), t.vxM(i.repeatNewPasswordControl.invalid && i.repeatNewPasswordControl.errors.notSame ? 42 : -1), t.R7$(), t.Y8G("disabled", i.passwordControl.invalid || i.newPasswordControl.invalid || i.repeatNewPasswordControl.invalid), t.R7$(2), t.SpI(" ", t.bMT(46, 36, "BTN_CHANGE"), " ")
                            }
                        },
                        dependencies: [h.Hu, h.RN, d.h, d.Mm, I.R, _.rl, _.nJ, _.MV, _.TL, T.fS, T.fg, l.YN, l.me, l.BC, l.X1, l.l_, b.Hl, b.$z, d.D9],
                        styles: ["\n\n\n\nmat-card[_ngcontent-%COMP%]{height:auto;min-width:320px;width:25%}mat-form-field[_ngcontent-%COMP%]{padding-top:10px}.form-container[_ngcontent-%COMP%]{display:flex;flex-direction:column;position:relative}button[_ngcontent-%COMP%]{margin-top:5px}#changeButton[_ngcontent-%COMP%]{margin-left:20%;margin-top:30px;width:60%}.mdc-card[_ngcontent-%COMP%]{border:0}em[_ngcontent-%COMP%]{color:var(--theme-text-fade-30)!important;display:flex;font-size:11px;white-space:nowrap}.mat-mdc-form-field-hint[_ngcontent-%COMP%]{align-items:center;color:var(--theme-text-fade-30)!important;display:flex;font-size:11px;margin-left:-10px!important;margin-right:-10px!important}.center-container[_ngcontent-%COMP%]{align-items:stretch;display:flex;justify-content:center}"]
                    })
                }
                return n
            })();

            function fr(n) {
                return function(e) {
                    return n.value !== e.value ? {
                        notSame: !0
                    } : null
                }
            }
            let ke = (() => {
                class n {
                    http;
                    hostServer = y.c.hostServer;
                    host = this.hostServer + "/api/Complaints";
                    constructor(e) {
                        this.http = e
                    }
                    save(e) {
                        return this.http.post(this.host + "/", e).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                            throw o
                        }))
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(s.KVO(E.Qq))
                    };
                    static \u0275prov = s.jDH({
                        token: n,
                        factory: n.\u0275fac,
                        providedIn: "root"
                    })
                }
                return n
            })();
            const _r = ["fileControl"],
                Cr = () => ({
                    length: "160"
                }),
                br = () => ({
                    type: "PDF, ZIP"
                }),
                vr = () => ({
                    size: "100 KB"
                });

            function kr(n, a) {
                1 & n && (t.j41(0, "div", 6), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.SpI("", t.i5U(2, 1, "INVALID_FILE_TYPE", t.lJ4(4, br)), " "))
            }

            function Sr(n, a) {
                1 & n && (t.j41(0, "div", 6), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.SpI("", t.i5U(2, 1, "INVALID_FILE_SIZE", t.lJ4(4, vr)), " "))
            }

            function Fr(n, a) {
                1 & n && (t.j41(0, "mat-error"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.SpI("", t.bMT(2, 1, "MANDATORY_MESSAGE"), " "))
            }
            S.Yv.add(C.iyL);
            let xr = (() => {
                    class n {
                        userService;
                        complaintService;
                        formSubmitService;
                        translate;
                        customerControl = new l.hs({
                            value: "",
                            disabled: !0
                        }, []);
                        messageControl = new l.hs("", [l.k0.required, l.k0.maxLength(160)]);
                        fileControl;
                        fileUploadError = void 0;
                        uploader = new Ft.l0({
                            url: y.c.hostServer + "/file-upload",
                            authToken: `Bearer ${localStorage.getItem("token")}`,
                            allowedMimeType: ["application/pdf", "application/xml", "text/xml", "application/zip", "application/x-zip-compressed", "multipart/x-zip", "application/yaml", "application/x-yaml", "text/yaml", "text/x-yaml"],
                            maxFileSize: 1e5
                        });
                        userEmail = void 0;
                        complaint = void 0;
                        confirmation;
                        constructor(e, o, i, r) {
                            this.userService = e, this.complaintService = o, this.formSubmitService = i, this.translate = r
                        }
                        ngOnInit() {
                            this.initComplaint(), this.uploader.onWhenAddingFileFailed = (e, o) => {
                                throw this.fileUploadError = o, new Error(`Error due to : ${o.name}`)
                            }, this.uploader.onAfterAddingFile = () => {
                                this.fileUploadError = void 0
                            }, this.uploader.onSuccessItem = () => {
                                this.saveComplaint(), this.uploader.clearQueue()
                            }, this.formSubmitService.attachEnterKeyHandler("complaint-form", "submitButton", () => {
                                this.save()
                            })
                        }
                        initComplaint() {
                            this.userService.whoAmI().subscribe({
                                next: e => {
                                    this.complaint = {}, this.complaint.UserId = e.id, this.userEmail = e.email, this.customerControl.setValue(this.userEmail)
                                },
                                error: e => {
                                    this.complaint = void 0, console.log(e)
                                }
                            })
                        }
                        save() {
                            this.uploader.queue[0] ? (this.uploader.queue[0].upload(), this.fileControl.nativeElement.value = null) : this.saveComplaint()
                        }
                        saveComplaint() {
                            this.complaint.message = this.messageControl.value, this.complaintService.save(this.complaint).subscribe({
                                next: e => {
                                    this.translate.get("CUSTOMER_SUPPORT_COMPLAINT_REPLY", {
                                        ref: e.id
                                    }).subscribe({
                                        next: o => {
                                            this.confirmation = o
                                        },
                                        error: o => {
                                            this.confirmation = o
                                        }
                                    }), this.initComplaint(), this.resetForm(), this.fileUploadError = void 0
                                },
                                error: e => e
                            })
                        }
                        resetForm() {
                            this.messageControl.setValue(""), this.messageControl.markAsUntouched(), this.messageControl.markAsPristine(), this.fileControl.nativeElement.value = null
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(P), t.rXU(ke), t.rXU(tt), t.rXU(d.c$))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-complaint"]
                            ],
                            viewQuery: function(o, i) {
                                if (1 & o && t.GBs(_r, 7), 2 & o) {
                                    let r;
                                    t.mGM(r = t.lsd()) && (i.fileControl = r.first)
                                }
                            },
                            decls: 44,
                            vars: 34,
                            consts: [
                                ["complaintMessage", ""],
                                ["fileControl", ""],
                                [1, "center-container"],
                                ["appearance", "outlined", 1, "mat-elevation-z6"],
                                [1, "mdc-card"],
                                [1, "confirmation", 3, "hidden"],
                                [1, "error", "fileUploadError"],
                                ["id", "complaint-form", 1, "form-container"],
                                ["appearance", "outline", "color", "accent"],
                                ["type", "text", "matInput", "", "aria-label", "Text field for the mail address of the user", 3, "formControl"],
                                [1, "fas", "fa-exclamation-circle"],
                                [2, "margin-left", "5px"],
                                ["id", "complaintMessage", "matAutosizeMinRows", "4", "matAutosizeMaxRows", "4", "matTextareaAutosize", "", "cols", "50", "maxlength", "160", "matInput", "", "aria-label", "Field for entering the complaint", 3, "formControl", "placeholder"],
                                ["align", "end"],
                                [1, "invoice"],
                                ["for", "file"],
                                ["ng2FileSelect", "", "id", "file", "type", "file", "accept", ".pdf,.zip", "aria-label", "Input area for uploading a single invoice PDF or XML B2B order file or a ZIP archive containing multiple invoices or orders\x3c!----\x3e", 2, "margin-left", "10px", 3, "uploader"],
                                ["type", "submit", "id", "submitButton", "mat-raised-button", "", "color", "primary", "aria-label", "Button to send the complaint", 3, "click", "disabled"]
                            ],
                            template: function(o, i) {
                                if (1 & o) {
                                    const r = t.RV6();
                                    t.j41(0, "div", 2)(1, "mat-card", 3)(2, "div", 4)(3, "h1"), t.EFF(4), t.nI1(5, "translate"), t.k0s(), t.j41(6, "div", 5), t.EFF(7), t.k0s(), t.nVh(8, kr, 3, 5, "div", 6), t.nVh(9, Sr, 3, 5, "div", 6), t.j41(10, "div", 7)(11, "mat-form-field", 8)(12, "mat-label"), t.EFF(13), t.nI1(14, "translate"), t.k0s(), t.nrm(15, "input", 9), t.k0s(), t.j41(16, "mat-form-field", 8)(17, "mat-label"), t.EFF(18), t.nI1(19, "translate"), t.k0s(), t.j41(20, "mat-hint"), t.nrm(21, "i", 10), t.j41(22, "em", 11), t.EFF(23), t.nI1(24, "translate"), t.k0s()(), t.nrm(25, "textarea", 12, 0), t.nI1(27, "translate"), t.nVh(28, Fr, 3, 3, "mat-error"), t.j41(29, "mat-hint", 13), t.EFF(30), t.k0s()(), t.j41(31, "div", 14)(32, "label", 15), t.EFF(33), t.nI1(34, "translate"), t.j41(35, "span"), t.EFF(36, ":"), t.k0s()(), t.nrm(37, "input", 16, 1), t.k0s()(), t.j41(39, "button", 17), t.bIt("click", function() {
                                        return s.eBV(r), s.Njj(i.save())
                                    }), t.j41(40, "mat-icon"), t.EFF(41, " send "), t.k0s(), t.EFF(42), t.nI1(43, "translate"), t.k0s()()()()
                                }
                                if (2 & o) {
                                    const r = t.sdS(26);
                                    t.R7$(4), t.JRh(t.bMT(5, 18, "NAV_COMPLAIN")), t.R7$(2), t.Y8G("hidden", !(i.confirmation && !i.messageControl.dirty)), t.R7$(), t.JRh(i.confirmation), t.R7$(), t.vxM(i.fileUploadError && "mimeType" == i.fileUploadError.name ? 8 : -1), t.R7$(), t.vxM(i.fileUploadError && "fileSize" == i.fileUploadError.name ? 9 : -1), t.R7$(4), t.JRh(t.bMT(14, 20, "LABEL_CUSTOMER")), t.R7$(2), t.Y8G("formControl", i.customerControl), t.R7$(3), t.JRh(t.bMT(19, 22, "LABEL_MESSAGE")), t.R7$(5), t.JRh(t.i5U(24, 24, "MAX_TEXTAREA_LENGTH", t.lJ4(33, Cr))), t.R7$(2), t.Y8G("placeholder", t.mNQ(t.bMT(27, 27, "WRITE_MESSAGE_PLACEHOLDER")))("formControl", i.messageControl), t.R7$(3), t.vxM(i.messageControl.invalid && null != i.messageControl && i.messageControl.errors.required ? 28 : -1), t.R7$(2), t.SpI("", (null == r.value ? null : r.value.length) || 0, "/160"), t.R7$(3), t.JRh(t.bMT(34, 29, "LABEL_INVOICE")), t.R7$(4), t.Y8G("uploader", i.uploader), t.R7$(2), t.Y8G("disabled", i.messageControl.invalid || i.fileUploadError), t.R7$(3), t.SpI(" ", t.bMT(43, 31, "BTN_SUBMIT"), " ")
                                }
                            },
                            dependencies: [h.Hu, h.RN, d.h, I.R, _.rl, _.nJ, _.MV, _.TL, T.fS, T.fg, l.YN, l.me, l.BC, l.tU, l.X1, l.l_, Ft.eB, Ft.UJ, b.Hl, b.$z, f.m_, f.An, d.D9],
                            styles: ["\n\n\n\nmat-card[_ngcontent-%COMP%]{height:auto;min-width:320px;width:30%}mat-form-field[_ngcontent-%COMP%]{padding-top:10px}.form-container[_ngcontent-%COMP%]{display:flex;flex-direction:column;position:relative}#submitButton[_ngcontent-%COMP%]{margin-left:20%;margin-top:30px;width:60%}.mdc-card[_ngcontent-%COMP%]{border:0}.invoice[_ngcontent-%COMP%]{display:flex;margin-left:3px;margin-top:15px}em[_ngcontent-%COMP%]{color:var(--theme-text-fade-30)!important;display:flex;font-size:11px;white-space:nowrap}.mat-mdc-form-field-hint[_ngcontent-%COMP%]{align-items:center;color:var(--theme-text-fade-30)!important;display:flex;font-size:11px;margin-left:-10px!important;margin-right:-10px!important}.center-container[_ngcontent-%COMP%]{align-items:stretch;display:flex;justify-content:center}"]
                        })
                    }
                    return n
                })(),
                Se = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/rest/chatbot";
                        constructor(e) {
                            this.http = e
                        }
                        getChatbotStatus() {
                            return this.http.get(this.host + "/status").pipe((0, v.T)(e => e), (0, g.W)(e => {
                                throw e
                            }))
                        }
                        getResponse(e, o) {
                            return this.http.post(this.host + "/respond", {
                                action: e,
                                query: o
                            }).pipe((0, v.T)(i => i), (0, g.W)(i => {
                                throw i
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })();
            const Tr = () => ({
                chatbot: "juicy-chat-bot"
            });

            function yr(n, a) {
                if (1 & n && t.nrm(0, "img", 16), 2 & n) {
                    const e = t.XpG(2);
                    t.Y8G("src", t.mNQ(e.juicyImageSrc), t.B4B)
                }
            }

            function Rr(n, a) {
                if (1 & n && t.nrm(0, "img", 17), 2 & n) {
                    const e = t.XpG(2);
                    t.Y8G("src", t.mNQ(e.profileImageSrc), t.B4B)
                }
            }

            function wr(n, a) {
                if (1 & n && (t.j41(0, "div", 10), t.nVh(1, yr, 1, 2, "img", 16), t.nVh(2, Rr, 1, 2, "img", 17), t.j41(3, "div"), t.EFF(4), t.k0s()()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.vxM("bot" == e.author ? 1 : -1), t.R7$(), t.vxM("user" == e.author ? 2 : -1), t.R7$(), t.HbH("user" == e.author ? "speech-bubble-right" : "speech-bubble-left"), t.R7$(), t.SpI(" ", e.body, " ")
                }
            }
            S.Yv.add(C.iyL);
            var ut = function(n) {
                return n.user = "user", n.bot = "bot", n
            }(ut || {});
            let Er = (() => {
                    class n {
                        userService;
                        chatbotService;
                        cookieService;
                        formSubmitService;
                        translate;
                        messageControl = new l.hs;
                        messages = [];
                        juicyImageSrc = "assets/public/images/ChatbotAvatar.png";
                        profileImageSrc = "assets/public/images/uploads/default.svg";
                        messageActions = {
                            response: "query",
                            namequery: "setname"
                        };
                        currentAction = this.messageActions.response;
                        chatScrollDownTimeoutId = null;
                        constructor(e, o, i, r, c) {
                            this.userService = e, this.chatbotService = o, this.cookieService = i, this.formSubmitService = r, this.translate = c
                        }
                        ngOnDestroy() {
                            this.chatScrollDownTimeoutId && clearTimeout(this.chatScrollDownTimeoutId)
                        }
                        ngOnInit() {
                            this.chatbotService.getChatbotStatus().subscribe(e => {
                                this.messages.push({
                                    author: ut.bot,
                                    body: e.body
                                }), e.action && (this.currentAction = this.messageActions[e.action])
                            }), this.userService.whoAmI().subscribe({
                                next: e => {
                                    this.profileImageSrc = e.profileImage
                                },
                                error: e => {
                                    console.log(e)
                                }
                            })
                        }
                        handleResponse(e) {
                            if (this.messages.push({
                                    author: ut.bot,
                                    body: e.body
                                }), this.currentAction = this.messageActions[e.action], e.token) {
                                localStorage.setItem("token", e.token);
                                const o = new Date;
                                o.setHours(o.getHours() + 8), this.cookieService.put("token", e.token, {
                                    expires: o
                                })
                            }
                        }
                        sendMessage() {
                            const e = this.messageControl.value;
                            e && (this.messages.push({
                                author: ut.user,
                                body: e
                            }), this.messageControl.setValue(""), this.chatbotService.getChatbotStatus().subscribe(o => {
                                o.status || o.action ? this.chatbotService.getResponse(this.currentAction, e).subscribe(i => {
                                    this.handleResponse(i)
                                }) : this.messages.push({
                                    author: ut.bot,
                                    body: o.body
                                }), this.chatScrollDownTimeoutId = setTimeout(() => {
                                    const i = document.getElementById("chat-window");
                                    i.scrollTop = i.scrollHeight, this.chatScrollDownTimeoutId = null
                                }, 250)
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(P), t.rXU(Se), t.rXU($.O1), t.rXU(tt), t.rXU(d.c$))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-chatbot"]
                            ],
                            decls: 24,
                            vars: 13,
                            consts: [
                                ["message", ""],
                                [1, "center-container"],
                                ["appearance", "outlined", 1, "mat-elevation-z6"],
                                [1, "mdc-card", "chat-wrapper"],
                                [1, "powered-by"],
                                ["appearance", "outlined", "id", "chat-container"],
                                [1, "mdc-card"],
                                ["appearance", "outlined", "id", "chat-box"],
                                [1, "mdc-card", "chat-content"],
                                ["id", "chat-window"],
                                [1, "message-container"],
                                [1, "message-box-container"],
                                [1, "form-wrapper"],
                                ["color", "accent", "appearance", "outline"],
                                ["translate", ""],
                                ["id", "message-input", "name", "message", "matInput", "", "aria-label", "Text field for a chat message", 3, "keyup.enter", "formControl", "placeholder"],
                                [1, "juicy-chat-bot-image", 3, "src"],
                                [1, "profile-image", 3, "src"]
                            ],
                            template: function(o, i) {
                                if (1 & o) {
                                    const r = t.RV6();
                                    t.j41(0, "div", 1)(1, "mat-card", 2)(2, "div", 3)(3, "h1"), t.EFF(4), t.nI1(5, "translate"), t.j41(6, "span", 4), t.EFF(7), t.nI1(8, "translate"), t.k0s()(), t.j41(9, "mat-card", 5)(10, "div", 6)(11, "mat-card", 7)(12, "div", 8)(13, "div", 9), t.Z7z(14, wr, 5, 5, "div", 10, t.fX1), t.k0s(), t.j41(16, "div", 11)(17, "div", 12)(18, "mat-form-field", 13)(19, "mat-label", 14), t.EFF(20, "LABEL_MESSAGE"), t.k0s(), t.j41(21, "input", 15, 0), t.nI1(23, "translate"), t.bIt("keyup.enter", function() {
                                        return s.eBV(r), s.Njj(i.sendMessage())
                                    }), t.k0s()()()()()()()()()()()
                                }
                                2 & o && (t.R7$(4), t.SpI("", t.bMT(5, 5, "SECTION_SUPPORT_CHAT"), " "), t.R7$(3), t.SpI("(", t.i5U(8, 7, "LABEL_POWERED_BY_CHATBOT", t.lJ4(12, Tr)), ")"), t.R7$(7), t.Dyx(i.messages), t.R7$(7), t.Y8G("placeholder", t.mNQ(t.bMT(23, 10, "ASK_ME_ANYTHING_PLACEHOLDER")))("formControl", i.messageControl))
                            },
                            dependencies: [h.Hu, h.RN, I.R, _.rl, _.nJ, d.h, d.Mm, T.fS, T.fg, l.YN, l.me, l.BC, l.X1, l.l_, d.D9],
                            styles: ['\n\n\n\nmat-card[_ngcontent-%COMP%]{height:auto;min-width:320px;width:60%}.chat-wrapper[_ngcontent-%COMP%]{display:flex;flex-direction:column;height:100%}.chat-content[_ngcontent-%COMP%]{display:flex;flex-direction:column;height:100%;position:relative}mat-form-field[_ngcontent-%COMP%]{display:block;margin:auto;width:70%}#chat-container[_ngcontent-%COMP%]{box-shadow:none;display:flex;position:relative;width:95%}#chat-box[_ngcontent-%COMP%]{background-color:#0000004d;display:flex;height:650px;margin-left:auto;margin-right:auto;padding:0;width:80%}#chat-window[_ngcontent-%COMP%]{flex:1;overflow-y:auto;padding:20px}#chat-window[_ngcontent-%COMP%]   [_ngcontent-%COMP%]::-webkit-scrollbar{background:none;width:20px}#chat-window[_ngcontent-%COMP%]   [_ngcontent-%COMP%]::-webkit-scrollbar-track{background:none}#chat-window[_ngcontent-%COMP%]   [_ngcontent-%COMP%]::-webkit-scrollbar-thumb{background:red;border-radius:10px}#chat-window[_ngcontent-%COMP%]   [_ngcontent-%COMP%]::-webkit-scrollbar-thumb:hover{background:#b30000}.message-box-container[_ngcontent-%COMP%]{border-top:solid 1px #fff;bottom:0;margin-bottom:-20px;position:sticky}.form-wrapper[_ngcontent-%COMP%]{font-size:10px;padding-top:15px}.message-container[_ngcontent-%COMP%]{height:auto;margin-right:25px;overflow:hidden;width:auto}.speech-bubble-right[_ngcontent-%COMP%]{background:#546e7a;border-radius:4px;float:right;font-size:15px;line-height:1.3;margin-bottom:25px;max-width:500px;padding:15px;position:relative}.speech-bubble-right[_ngcontent-%COMP%]:after{border-left:20px solid transparent;border-top:20px solid rgb(84,110,122);bottom:-20px;content:"";position:absolute;right:20px}.speech-bubble-left[_ngcontent-%COMP%]{background:#546e7a;border-radius:4px;float:left;font-size:15px;line-height:1.3;margin-bottom:25px;max-width:500px;padding:15px;position:relative}.speech-bubble-left[_ngcontent-%COMP%]:after{border-right:20px solid transparent;border-top:20px solid rgb(84,110,122);bottom:-20px;content:"";left:20px;position:absolute}.juicy-chat-bot-image[_ngcontent-%COMP%]{float:left;height:70px;margin-top:20px}.profile-image[_ngcontent-%COMP%]{border-radius:50%;float:right;height:40px;margin-left:10px;margin-top:20px}.powered-by[_ngcontent-%COMP%]{font-size:xx-small}.center-container[_ngcontent-%COMP%]{align-items:stretch;display:flex;justify-content:center}']
                        })
                    }
                    return n
                })(),
                Fe = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/api/Recycles";
                        constructor(e) {
                            this.http = e
                        }
                        find(e) {
                            return this.http.get(this.host + "/", {
                                params: e
                            }).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        save(e) {
                            return this.http.post(this.host + "/", e).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })(),
                st = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/api/Addresss";
                        constructor(e) {
                            this.http = e
                        }
                        get() {
                            return this.http.get(this.host).pipe((0, v.T)(e => e.data), (0, g.W)(e => {
                                throw e
                            }))
                        }
                        getById(e) {
                            return this.http.get(`${this.host}/${e}`).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        save(e) {
                            return this.http.post(this.host + "/", e).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        put(e, o) {
                            return this.http.put(`${this.host}/${e}`, o).pipe((0, v.T)(i => i.data), (0, g.W)(i => {
                                throw i
                            }))
                        }
                        del(e) {
                            return this.http.delete(`${this.host}/${e}`).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })();
            var xe = m(3869);

            function Mr(n, a) {
                1 & n && (t.j41(0, "h1"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.JRh(t.bMT(2, 1, "TITLE_SELECT_ADDRESS")))
            }

            function Ir(n, a) {
                1 & n && (t.j41(0, "h1"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.JRh(t.bMT(2, 1, "MY_SAVED_ADRESSES")))
            }

            function Or(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 15)
            }

            function Pr(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-cell")(1, "mat-radio-button", 16), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG(2);
                        return s.Njj(r.emitSelectionToParent(i.id))
                    })("change", function(i) {
                        const r = s.eBV(e).$implicit,
                            c = t.XpG(2);
                        return s.Njj(i ? c.selection.toggle(r) : null)
                    }), t.k0s()()
                }
                if (2 & n) {
                    const e = a.$implicit,
                        o = t.XpG(2);
                    t.R7$(), t.Y8G("checked", o.selection.isSelected(e))
                }
            }

            function jr(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 15)
            }

            function Ar(n, a) {
                if (1 & n && (t.j41(0, "mat-cell"), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.SpI(" ", null == e ? null : e.fullName, " ")
                }
            }

            function Dr(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 15)
            }

            function $r(n, a) {
                if (1 & n && (t.j41(0, "mat-cell"), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.LHq(" ", null == e ? null : e.streetAddress, ", ", null == e ? null : e.city, ", ", null == e ? null : e.state, ", ", null == e ? null : e.zipCode, " ")
                }
            }

            function Lr(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 15)
            }

            function Br(n, a) {
                if (1 & n && (t.j41(0, "mat-cell"), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.SpI(" ", null == e ? null : e.country, " ")
                }
            }

            function Nr(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 15)
            }

            function Ur(n, a) {
                if (1 & n && (t.j41(0, "mat-cell")(1, "button", 17), t.nrm(2, "i", 18), t.k0s()()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.Y8G("routerLink", t.VkB("/address/edit/", e.id))
                }
            }

            function Vr(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 15)
            }

            function Gr(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-cell")(1, "button", 19), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG(2);
                        return s.Njj(r.deleteAddress(i.id))
                    }), t.nrm(2, "i", 20), t.k0s()()
                }
            }

            function Yr(n, a) {
                1 & n && t.nrm(0, "mat-header-row")
            }

            function Hr(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-row", 21), t.bIt("click", function() {
                        const i = s.eBV(e),
                            r = i.$implicit,
                            c = i.$implicit,
                            p = t.XpG(2);
                        return p.selection.toggle(r), s.Njj(p.emitSelectionToParent(c.id))
                    }), t.k0s()
                }
            }

            function Xr(n, a) {
                if (1 & n && (t.j41(0, "mat-table", 2), t.qex(1, 5), t.DNE(2, Or, 1, 0, "mat-header-cell", 6)(3, Pr, 2, 1, "mat-cell", 7), t.bVm(), t.qex(4, 8), t.DNE(5, jr, 1, 0, "mat-header-cell", 6)(6, Ar, 2, 1, "mat-cell", 7), t.bVm(), t.qex(7, 9), t.DNE(8, Dr, 1, 0, "mat-header-cell", 6)(9, $r, 2, 4, "mat-cell", 7), t.bVm(), t.qex(10, 10), t.DNE(11, Lr, 1, 0, "mat-header-cell", 6)(12, Br, 2, 1, "mat-cell", 7), t.bVm(), t.qex(13, 11), t.DNE(14, Nr, 1, 0, "mat-header-cell", 6)(15, Ur, 3, 2, "mat-cell", 7), t.bVm(), t.qex(16, 12), t.DNE(17, Vr, 1, 0, "mat-header-cell", 6)(18, Gr, 3, 0, "mat-cell", 7), t.bVm(), t.DNE(19, Yr, 1, 0, "mat-header-row", 13)(20, Hr, 1, 0, "mat-row", 14), t.k0s()), 2 & n) {
                    const e = t.XpG();
                    t.Y8G("dataSource", e.dataSource), t.R7$(19), t.Y8G("matHeaderRowDef", e.displayedColumns), t.R7$(), t.Y8G("matRowDefColumns", e.displayedColumns)
                }
            }

            function Wr(n, a) {
                1 & n && (t.j41(0, "div", 3)(1, "button", 22)(2, "mat-icon"), t.EFF(3, "add"), t.k0s(), t.j41(4, "span"), t.EFF(5), t.nI1(6, "translate"), t.k0s()()()), 2 & n && (t.R7$(5), t.JRh(t.bMT(6, 1, "ADD_NEW_ADDRESS")))
            }

            function zr(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 23), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.chooseAddress())
                    }), t.j41(1, "mat-icon"), t.EFF(2, "navigate_next"), t.k0s(), t.j41(3, "span", 24), t.EFF(4), t.nI1(5, "translate"), t.k0s()()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.Y8G("disabled", void 0 === e.addressId), t.R7$(4), t.JRh(t.bMT(5, 2, "LABEL_CONTINUE"))
                }
            }
            S.Yv.add(H.MT, H.UB);
            let Wt = (() => {
                class n {
                    addressService;
                    translate;
                    router;
                    ngZone;
                    snackBarHelperService;
                    emitSelection = new t.bkB;
                    allowEdit = !1;
                    addNewAddressDiv = !0;
                    showNextButton = !1;
                    addressId = void 0;
                    displayedColumns = ["Name", "Address", "Country"];
                    selection = new xe.C(!1, []);
                    storedAddresses;
                    dataSource;
                    confirmation;
                    error;
                    addressExist = !1;
                    constructor(e, o, i, r, c) {
                        this.addressService = e, this.translate = o, this.router = i, this.ngZone = r, this.snackBarHelperService = c
                    }
                    ngOnInit() {
                        this.allowEdit ? this.displayedColumns.push("Edit", "Remove") : this.displayedColumns.unshift("Selection"), this.load()
                    }
                    load() {
                        this.addressService.get().subscribe({
                            next: e => {
                                this.addressExist = e.length, this.storedAddresses = e, this.dataSource = new u.I6(this.storedAddresses)
                            },
                            error: e => {
                                this.snackBarHelperService.open(e.error?.error, "errorBar"), console.log(e)
                            }
                        })
                    }
                    emitSelectionToParent(e) {
                        this.selection.hasValue() ? (this.emitSelection.emit(e), this.addressId = e) : (this.emitSelection.emit(void 0), this.addressId = void 0)
                    }
                    chooseAddress() {
                        var e = this;
                        sessionStorage.setItem("addressId", this.addressId), this.ngZone.run((0, F.A)(function*() {
                            return yield e.router.navigate(["/delivery-method"])
                        }))
                    }
                    deleteAddress(e) {
                        this.addressService.del(e).subscribe({
                            next: () => {
                                this.error = null, this.translate.get("ADDRESS_REMOVED").subscribe({
                                    next: o => {
                                        this.snackBarHelperService.open(o, "confirmBar")
                                    },
                                    error: o => {
                                        this.snackBarHelperService.open(o, "confirmBar")
                                    }
                                }), this.load()
                            },
                            error: o => {
                                this.snackBarHelperService.open(o.error?.error, "errorBar"), console.log(o)
                            }
                        })
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(st), t.rXU(d.c$), t.rXU(R.Ix), t.rXU(t.SKi), t.rXU(L.r))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-address"]
                        ],
                        inputs: {
                            allowEdit: "allowEdit",
                            addNewAddressDiv: "addNewAddressDiv",
                            showNextButton: "showNextButton"
                        },
                        outputs: {
                            emitSelection: "emitSelection"
                        },
                        decls: 7,
                        vars: 6,
                        consts: [
                            ["appearance", "outlined", 1, "mat-elevation-z6"],
                            [1, "mdc-card"],
                            [1, "address-table", 3, "dataSource"],
                            [1, "add-new-address"],
                            ["mat-raised-button", "", "mat-button", "", "color", "primary", "aria-label", "Proceed to payment selection", 1, "btn", "btn-next", 3, "disabled"],
                            ["matColumnDef", "Selection"],
                            ["style", "display: none;", 4, "matHeaderCellDef"],
                            [4, "matCellDef"],
                            ["matColumnDef", "Name"],
                            ["matColumnDef", "Address"],
                            ["matColumnDef", "Country"],
                            ["matColumnDef", "Edit"],
                            ["matColumnDef", "Remove"],
                            [4, "matHeaderRowDef"],
                            [3, "click", 4, "matRowDef", "matRowDefColumns"],
                            [2, "display", "none"],
                            [3, "click", "change", "checked"],
                            ["mat-icon-button", "", 3, "routerLink"],
                            [1, "far", "fa-edit"],
                            ["mat-icon-button", "", 3, "click"],
                            [1, "far", "fa-trash-alt"],
                            [3, "click"],
                            ["mat-raised-button", "", "mat-button", "", "color", "primary", "aria-label", "Add a new address", "routerLink", "/address/create", 1, "btn", "btn-new-address"],
                            ["mat-raised-button", "", "mat-button", "", "color", "primary", "aria-label", "Proceed to payment selection", 1, "btn", "btn-next", 3, "click", "disabled"],
                            ["translate", ""]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "mat-card", 0)(1, "div", 1), t.nVh(2, Mr, 3, 3, "h1")(3, Ir, 3, 3, "h1"), t.nVh(4, Xr, 21, 3, "mat-table", 2), t.nVh(5, Wr, 7, 3, "div", 3), t.nVh(6, zr, 6, 4, "button", 4), t.k0s()()), 2 & o && (t.AVh("div-boundary", !i.addNewAddressDiv), t.R7$(2), t.vxM(i.showNextButton ? 2 : 3), t.R7$(2), t.vxM(i.addressExist ? 4 : -1), t.R7$(), t.vxM(i.addNewAddressDiv ? 5 : -1), t.R7$(), t.vxM(i.showNextButton ? 6 : -1))
                        },
                        dependencies: [h.Hu, h.RN, d.h, d.Mm, u.Zl, u.cC, u.tL, u.KS, u.YV, u.$R, ot._g, V.iY, nt.Wk, u.ji, u.YZ, u.iL, u.NB, b.Hl, b.$z, f.m_, f.An, d.D9],
                        styles: ["\n\n\n\n.btn-new-address[_ngcontent-%COMP%]{display:block;margin-left:0;margin-right:auto}.btn-next[_ngcontent-%COMP%]{display:block;margin-left:auto;margin-right:0;margin-top:-37px}mat-header-row[_ngcontent-%COMP%]{min-height:0!important}.heading[_ngcontent-%COMP%]{background:#0003;font-size:x-large}.heading-text[_ngcontent-%COMP%]{justify-content:center;padding:12px 20px}.div-boundary[_ngcontent-%COMP%]{border:1px solid rgba(255,255,255,.3);border-radius:5px;box-shadow:0 0!important}.mdc-card[_ngcontent-%COMP%]{border:0}mat-table[_ngcontent-%COMP%]{border-bottom:1px solid var(--theme-background-light);border-top:1px solid var(--theme-background-light)}mat-cell[_ngcontent-%COMP%]{padding:0!important}svg[_ngcontent-%COMP%]{font-size:12.5px}.address-table[_ngcontent-%COMP%]{width:100%}.mat-column-Selection[_ngcontent-%COMP%], .mat-column-Name[_ngcontent-%COMP%]{width:20%}.mat-column-Address[_ngcontent-%COMP%]{width:40%}.mat-column-Country[_ngcontent-%COMP%]{width:20%}.mat-column-Edit[_ngcontent-%COMP%], .mat-column-Remove[_ngcontent-%COMP%]{text-align:right;width:48px}.mat-column-Address[_ngcontent-%COMP%]{word-break:break-word}.add-new-address[_ngcontent-%COMP%]{margin-top:20px}"]
                    })
                }
                return n
            })();
            const Jr = ["addressComp"],
                Qr = () => ({
                    range: "10-1000"
                });

            function qr(n, a) {
                1 & n && (t.j41(0, "mat-error", 8), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.SpI(" ", t.bMT(2, 1, "MANDATORY_QUANTITY"), " "))
            }

            function Kr(n, a) {
                1 & n && (t.j41(0, "mat-error"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.SpI(" ", t.i5U(2, 1, "INVALID_QUANTITY", t.lJ4(4, Qr)), " "))
            }

            function Zr(n, a) {
                1 & n && (t.j41(0, "mat-error"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.JRh(t.bMT(2, 1, "INVALID_DATE")))
            }

            function ts(n, a) {
                if (1 & n && (t.j41(0, "mat-form-field", 10)(1, "mat-label"), t.EFF(2), t.nI1(3, "translate"), t.k0s(), t.nrm(4, "input", 20)(5, "mat-datepicker-toggle", 21)(6, "mat-datepicker", null, 1), t.nVh(8, Zr, 3, 3, "mat-error"), t.k0s()), 2 & n) {
                    const e = t.sdS(7),
                        o = t.XpG();
                    t.R7$(2), t.JRh(t.bMT(3, 5, "LABEL_PICKUP_DATE")), t.R7$(2), t.Y8G("formControl", o.pickUpDateControl)("matDatepicker", e), t.R7$(), t.Y8G("for", e), t.R7$(3), t.vxM(o.pickUpDateControl.invalid ? 8 : -1)
                }
            }

            function es(n, a) {
                if (1 & n && (t.j41(0, "mat-checkbox", 11), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n) {
                    const e = t.XpG();
                    t.Y8G("formControl", e.pickup), t.R7$(), t.JRh(t.bMT(2, 2, "REQUEST_PICKUP"))
                }
            }
            S.Yv.add(C.isI);
            let ns = (() => {
                    class n {
                        recycleService;
                        userService;
                        configurationService;
                        formSubmitService;
                        translate;
                        snackBarHelperService;
                        addressComponent;
                        requestorControl = new l.hs({
                            value: "",
                            disabled: !0
                        }, []);
                        recycleQuantityControl = new l.hs("", [l.k0.required, l.k0.min(10), l.k0.max(1e3)]);
                        pickUpDateControl = new l.hs;
                        pickup = new l.hs(!1);
                        topImage;
                        bottomImage;
                        recycles;
                        recycle = {};
                        userEmail;
                        confirmation;
                        addressId = void 0;
                        constructor(e, o, i, r, c, p) {
                            this.recycleService = e, this.userService = o, this.configurationService = i, this.formSubmitService = r, this.translate = c, this.snackBarHelperService = p
                        }
                        ngOnInit() {
                            this.configurationService.getApplicationConfiguration().subscribe({
                                next: e => {
                                    e?.application?.recyclePage && (this.topImage = `assets/public/images/products/${e.application.recyclePage.topProductImage}`, this.bottomImage = `assets/public/images/products/${e.application.recyclePage.bottomProductImage}`)
                                },
                                error: e => {
                                    console.log(e)
                                }
                            }), this.initRecycle(), this.findAll(), this.formSubmitService.attachEnterKeyHandler("recycle-form", "recycleButton", () => {
                                this.save()
                            })
                        }
                        initRecycle() {
                            this.userService.whoAmI().subscribe({
                                next: e => {
                                    this.recycle = {}, this.recycle.UserId = e.id, this.userEmail = e.email, this.requestorControl.setValue(this.userEmail)
                                },
                                error: e => {
                                    console.log(e)
                                }
                            })
                        }
                        save() {
                            this.recycle.AddressId = this.addressId, this.recycle.quantity = this.recycleQuantityControl.value, this.pickup.value && (this.recycle.isPickUp = this.pickup.value, this.recycle.date = this.pickUpDateControl.value), this.recycleService.save(this.recycle).subscribe({
                                next: e => {
                                    e.isPickup ? this.translate.get("CONFIRM_RECYCLING_PICKUP", {
                                        pickupdate: e.pickupDate
                                    }).subscribe({
                                        next: o => {
                                            this.snackBarHelperService.open(o, "confirmBar")
                                        },
                                        error: o => {
                                            this.snackBarHelperService.open(o, "confirmBar")
                                        }
                                    }) : this.translate.get("CONFIRM_RECYCLING_BOX").subscribe({
                                        next: o => {
                                            this.snackBarHelperService.open(o, "confirmBar")
                                        },
                                        error: o => {
                                            this.snackBarHelperService.open(o, "confirmBar")
                                        }
                                    }), this.addressComponent.load(), this.initRecycle(), this.resetForm()
                                },
                                error: e => {
                                    this.snackBarHelperService.open(e.error?.error, "errorBar"), console.log(e)
                                }
                            })
                        }
                        findAll() {
                            this.recycleService.find().subscribe({
                                next: e => {
                                    this.recycles = e
                                },
                                error: e => {
                                    console.log(e)
                                }
                            })
                        }
                        resetForm() {
                            this.addressId = void 0, this.recycleQuantityControl.setValue(""), this.recycleQuantityControl.markAsPristine(), this.recycleQuantityControl.markAsUntouched(), this.pickUpDateControl.setValue(""), this.pickUpDateControl.markAsPristine(), this.pickUpDateControl.markAsUntouched(), this.pickup.setValue(!1)
                        }
                        getMessage(e) {
                            this.addressId = e
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(Fe), t.rXU(P), t.rXU(M.s), t.rXU(tt), t.rXU(d.c$), t.rXU(L.r))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-recycle"]
                            ],
                            viewQuery: function(o, i) {
                                if (1 & o && t.GBs(Jr, 7), 2 & o) {
                                    let r;
                                    t.mGM(r = t.lsd()) && (i.addressComponent = r.first)
                                }
                            },
                            decls: 45,
                            vars: 28,
                            consts: [
                                ["addressComp", ""],
                                ["picker", ""],
                                [1, "mat-elevation-z6", "mat-own-card", "recycle-container"],
                                [1, "left-container", "flex-50"],
                                ["id", "recycle-form", 1, "form-container"],
                                ["appearance", "outline", "color", "accent"],
                                ["type", "text", "matInput", "", 3, "formControl"],
                                ["type", "number", "matInput", "", 3, "formControl", "placeholder"],
                                ["translate", ""],
                                [1, "mat-elevation-z0", 3, "emitSelection", "addNewAddressDiv"],
                                ["appearance", "outline"],
                                [3, "formControl"],
                                ["type", "submit", "id", "recycleButton", "mat-raised-button", "", "color", "primary", 3, "click", "disabled"],
                                [1, "fas", "fa-paper-plane", "fa-lg"],
                                [1, "right-container", "flex-50"],
                                [1, "right-inner"],
                                [1, "responsibility-header"],
                                [1, "mat-elevation-z0", "card-row"],
                                ["mat-card-image", "", 3, "src"],
                                [1, "fill-remaining-space"],
                                ["matInput", "", 3, "formControl", "matDatepicker"],
                                ["matSuffix", "", 3, "for"]
                            ],
                            template: function(o, i) {
                                if (1 & o) {
                                    const r = t.RV6();
                                    t.j41(0, "mat-card", 2)(1, "div", 3)(2, "h1"), t.EFF(3), t.nI1(4, "translate"), t.k0s(), t.j41(5, "div", 4)(6, "mat-form-field", 5)(7, "mat-label"), t.EFF(8), t.nI1(9, "translate"), t.k0s(), t.nrm(10, "input", 6), t.k0s(), t.j41(11, "mat-form-field", 5)(12, "mat-label"), t.EFF(13), t.nI1(14, "translate"), t.k0s(), t.nrm(15, "input", 7), t.nI1(16, "translate"), t.nVh(17, qr, 3, 3, "mat-error", 8), t.nVh(18, Kr, 3, 5, "mat-error"), t.k0s(), t.j41(19, "app-address", 9, 0), t.bIt("emitSelection", function(p) {
                                        return s.eBV(r), s.Njj(i.getMessage(p))
                                    }), t.k0s(), t.nVh(21, ts, 9, 7, "mat-form-field", 10), t.nVh(22, es, 3, 4, "mat-checkbox", 11), t.k0s(), t.j41(23, "button", 12), t.bIt("click", function() {
                                        return s.eBV(r), s.Njj(i.save())
                                    }), t.nrm(24, "i", 13), t.EFF(25), t.nI1(26, "translate"), t.k0s()(), t.j41(27, "div", 14)(28, "div", 15)(29, "h3", 16), t.EFF(30), t.nI1(31, "translate"), t.k0s(), t.j41(32, "mat-card", 17), t.nrm(33, "img", 18), t.j41(34, "mat-card-content")(35, "div")(36, "small"), t.EFF(37, "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. "), t.k0s()()()(), t.j41(38, "mat-card", 17), t.nrm(39, "img", 18), t.j41(40, "mat-card-content")(41, "div")(42, "small"), t.EFF(43, "Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. "), t.k0s()()()(), t.nrm(44, "span", 19), t.k0s()()()
                                }
                                2 & o && (t.R7$(3), t.JRh(t.bMT(4, 16, "TITLE_RECYCLE")), t.R7$(5), t.JRh(t.bMT(9, 18, "LABEL_REQUESTOR")), t.R7$(2), t.Y8G("formControl", i.requestorControl), t.R7$(3), t.JRh(t.bMT(14, 20, "LABEL_QUANTITY")), t.R7$(2), t.Y8G("formControl", i.recycleQuantityControl)("placeholder", t.bMT(16, 22, "IN_LITERS_PLACEHOLDER")), t.R7$(2), t.vxM(i.recycleQuantityControl.invalid && i.recycleQuantityControl.errors.required ? 17 : -1), t.R7$(), t.vxM(i.recycleQuantityControl.invalid && (i.recycleQuantityControl.errors.min || i.recycleQuantityControl.errors.max) ? 18 : -1), t.R7$(), t.Y8G("addNewAddressDiv", !1), t.R7$(2), t.vxM(i.pickup.value && i.recycleQuantityControl.value > 100 ? 21 : -1), t.R7$(), t.vxM(i.recycleQuantityControl.value > 100 ? 22 : -1), t.R7$(), t.Y8G("disabled", void 0 === i.addressId || i.recycleQuantityControl.invalid || i.pickUpDateControl.invalid), t.R7$(2), t.SpI(" ", t.bMT(26, 24, "BTN_SUBMIT"), " "), t.R7$(5), t.JRh(t.bMT(31, 26, "SECTION_PRESS_JUICE_RESPONSIBLY")), t.R7$(3), t.Y8G("src", i.topImage, t.B4B), t.R7$(6), t.Y8G("src", i.bottomImage, t.B4B))
                            },
                            dependencies: [h.Hu, h.RN, h.m2, h.kF, d.h, d.Mm, I.R, _.rl, _.nJ, _.TL, _.yw, T.fS, T.fg, l.YN, l.me, l.Q0, l.BC, l.X1, l.l_, Wt, kt.bZ, kt.bU, kt.Vh, Nt.So, b.Hl, b.$z, d.D9],
                            styles: ["\n\n\n\nmat-form-field[_ngcontent-%COMP%]{padding-top:10px}.form-container[_ngcontent-%COMP%]{display:flex;flex-direction:column;position:relative}button[_ngcontent-%COMP%]{margin-left:20%;margin-top:5px;width:60%}.mat-own-card[_ngcontent-%COMP%]{display:block;margin-left:10%;margin-right:10%}mat-card[_ngcontent-%COMP%]   img[_ngcontent-%COMP%]{height:100px;margin-left:15px!important;margin-right:2px!important}.responsibility-header[_ngcontent-%COMP%]{margin-left:15px;margin-top:10px}#recycle-form[_ngcontent-%COMP%]{margin-left:15px;margin-right:15px}  .mat-mdc-row.mdc-data-table__row{border-bottom:1px solid var(--theme-background-light)!important;border-top:1px solid var(--theme-background-light)!important}h1[_ngcontent-%COMP%]{margin-left:15px;margin-top:10px}.mat-mdc-card-content[_ngcontent-%COMP%]{margin-right:20px}.left-container[_ngcontent-%COMP%]{display:flex;flex-direction:column;gap:20px;margin-bottom:20px!important}.fill-remaining-space[_ngcontent-%COMP%]{flex:1 1 auto}.recycle-container[_ngcontent-%COMP%]{display:flex;flex-direction:row}@media (max-width: 959.98px){.recycle-container[_ngcontent-%COMP%]{flex-direction:column;gap:20px}.flex-50[_ngcontent-%COMP%]{flex:0 0 100%;max-width:100%}}.right-container[_ngcontent-%COMP%]{display:flex;flex-direction:column;gap:20px}.right-inner[_ngcontent-%COMP%]{align-items:center;display:flex;flex-direction:column;gap:20px}.card-row[_ngcontent-%COMP%]{align-items:flex-start;display:flex;flex-direction:row;gap:20px}.flex-50[_ngcontent-%COMP%]{flex:0 0 50%;max-width:50%}"]
                        })
                    }
                    return n
                })(),
                Te = (() => {
                    class n {
                        http;
                        constructor(e) {
                            this.http = e
                        }
                        verify(e) {
                            return this.http.post(`${y.c.hostServer}/rest/2fa/verify`, {
                                tmpToken: localStorage.getItem("totp_tmp_token"),
                                totpToken: e
                            }).pipe((0, v.T)(o => o.authentication), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        status() {
                            return this.http.get(`${y.c.hostServer}/rest/2fa/status`).pipe((0, v.T)(e => e), (0, g.W)(e => {
                                throw e
                            }))
                        }
                        setup(e, o, i) {
                            return this.http.post(`${y.c.hostServer}/rest/2fa/setup`, {
                                password: e,
                                setupToken: i,
                                initialToken: o
                            }).pipe((0, v.T)(() => {}), (0, g.W)(r => {
                                throw r
                            }))
                        }
                        disable(e) {
                            return this.http.post(`${y.c.hostServer}/rest/2fa/disable`, {
                                password: e
                            }).pipe((0, v.T)(() => {}), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })();

            function os(n, a) {
                1 & n && (t.j41(0, "div", 5), t.EFF(1, "INVALID_TWO_FACTOR_AUTH_TOKEN"), t.k0s())
            }
            S.Yv.add(C.HEq);
            let is = (() => {
                class n {
                    twoFactorAuthService;
                    cookieService;
                    userService;
                    router;
                    ngZone;
                    twoFactorForm = new l.J3({
                        token: new l.hs("", [l.k0.minLength(6), l.k0.maxLength(6), l.k0.required, l.k0.pattern("^[\\d]{6}$")])
                    });
                    errored = !1;
                    constructor(e, o, i, r, c) {
                        this.twoFactorAuthService = e, this.cookieService = o, this.userService = i, this.router = r, this.ngZone = c
                    }
                    verify() {
                        var e = this;
                        this.twoFactorAuthService.verify(this.twoFactorForm.value.token).subscribe({
                            next: i => {
                                localStorage.setItem("token", i.token);
                                const r = new Date;
                                r.setHours(r.getHours() + 8), this.cookieService.put("token", i.token, {
                                    expires: r
                                }), sessionStorage.setItem("bid", i.bid?.toString()), this.userService.isLoggedIn.next(!0), this.ngZone.run((0, F.A)(function*() {
                                    return yield e.router.navigate(["/search"])
                                }))
                            },
                            error: i => (this.errored = !0, setTimeout(() => {
                                this.errored = !1
                            }, 5e3), i)
                        })
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(Te), t.rXU($.O1), t.rXU(P), t.rXU(R.Ix), t.rXU(t.SKi))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-two-factor-auth-enter"]
                        ],
                        decls: 28,
                        vars: 15,
                        consts: [
                            ["tokenInput", ""],
                            [1, "center-container"],
                            ["appearance", "outlined", 1, "mat-elevation-z6"],
                            [1, "mdc-card"],
                            ["translate", ""],
                            ["translate", "", 1, "error", 2, "margin-bottom", "10px"],
                            [3, "ngSubmit", "formGroup"],
                            [1, "form-container"],
                            ["id", "inputToken", "appearance", "outline", "color", "accent"],
                            ["formControlName", "token", "type", "text", "minlength", "6", "maxlength", "6", "pattern", "^[\\d]{6}$", "matInput", "", "id", "totpToken", "aria-label", "Field for entering the Two Factor token", 3, "placeholder"],
                            ["matSuffix", "", "matTooltipPosition", "right", "aria-label", "The code to be entered from the authenticator must have 6 digits.", 3, "matTooltip"],
                            ["align", "end"],
                            ["type", "submit", "color", "primary", "mat-raised-button", "", "id", "totpSubmitButton", "aria-label", "Button to confirm the input", 3, "disabled"]
                        ],
                        template: function(o, i) {
                            if (1 & o) {
                                const r = t.RV6();
                                t.j41(0, "div", 1)(1, "mat-card", 2)(2, "div", 3)(3, "h1", 4), t.EFF(4, "TITLE_TWO_FACTOR_AUTHENTICATION"), t.k0s(), t.j41(5, "p", 4), t.EFF(6, "TITLE_TWO_FACTOR_AUTH_ENTER"), t.k0s(), t.nVh(7, os, 2, 0, "div", 5), t.j41(8, "form", 6), t.bIt("ngSubmit", function() {
                                    return s.eBV(r), s.Njj(i.verify())
                                }), t.j41(9, "div", 7)(10, "mat-form-field", 8)(11, "mat-label", 4), t.EFF(12, "LABEL_TWO_FACTOR_AUTH_TOKEN"), t.k0s(), t.nrm(13, "input", 9, 0), t.nI1(15, "translate"), t.j41(16, "mat-icon", 10), t.nI1(17, "translate"), t.EFF(18, " help_outline "), t.k0s(), t.j41(19, "mat-hint", 11), t.EFF(20), t.k0s(), t.j41(21, "mat-error", 4), t.EFF(22, "INVALID_TWO_FACTOR_AUTH_TOKEN"), t.k0s()()(), t.j41(23, "button", 12)(24, "mat-icon"), t.EFF(25, "lock_open"), t.k0s(), t.EFF(26), t.nI1(27, "translate"), t.k0s()()()()()
                            }
                            if (2 & o) {
                                const r = t.sdS(14);
                                t.R7$(7), t.vxM(i.errored ? 7 : -1), t.R7$(), t.Y8G("formGroup", i.twoFactorForm), t.R7$(5), t.Y8G("placeholder", t.mNQ(t.bMT(15, 9, "2FA_ENTER_CODE_PLACEHOLDER"))), t.R7$(3), t.Y8G("matTooltip", t.mNQ(t.bMT(17, 11, "INITIAL_TOKEN_TOOLTIP"))), t.R7$(4), t.SpI("", (null == r.value ? null : r.value.length) || 0, "/6"), t.R7$(3), t.Y8G("disabled", i.twoFactorForm.invalid), t.R7$(3), t.SpI(" ", t.bMT(27, 13, "BTN_LOGIN"), " ")
                            }
                        },
                        dependencies: [h.Hu, h.RN, d.h, d.Mm, l.YN, l.qT, l.me, l.BC, l.cb, l.xh, l.tU, l.R_, l.X1, l.j4, l.JD, I.R, _.rl, _.nJ, _.MV, _.TL, _.yw, T.fS, T.fg, f.m_, f.An, B.oV, b.Hl, b.$z, d.D9],
                        styles: ["\n\n\n\nmat-card[_ngcontent-%COMP%]{height:auto;min-width:320px;width:25%}#totpSubmitButton[_ngcontent-%COMP%]{margin-left:20%;margin-top:30px;width:60%}#inputToken[_ngcontent-%COMP%]{width:100%}.center-container[_ngcontent-%COMP%]{align-items:stretch;display:flex;justify-content:center}"]
                    })
                }
                return n
            })();
            S.Yv.add(C.c7y, C.JUx);
            let as = (() => {
                    class n {
                        route;
                        translate;
                        error = null;
                        constructor(e, o) {
                            this.route = e, this.translate = o
                        }
                        ngOnInit() {
                            const e = this.route.snapshot.queryParams.error;
                            e && this.translate.get(e).subscribe({
                                next: o => {
                                    this.error = o
                                },
                                error: o => {
                                    this.error = o
                                }
                            })
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(R.nX), t.rXU(d.c$))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-error-page"]
                            ],
                            decls: 7,
                            vars: 1,
                            consts: [
                                ["appearance", "outlined", 1, "warn-notification", "mat-elevation-z6"],
                                [1, "mdc-card"],
                                [1, "mat-headline-2"],
                                [1, "fas", "fa-hand-paper"],
                                [1, "mat-subtitle-1"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "mat-card", 0)(1, "div", 1)(2, "div", 2), t.nrm(3, "i", 3), t.EFF(4, " 403"), t.k0s(), t.j41(5, "span", 4), t.EFF(6), t.k0s()()()), 2 & o && (t.R7$(6), t.JRh(i.error))
                            },
                            dependencies: [h.Hu, h.RN],
                            styles: ["\n\n\n\nmat-card[_ngcontent-%COMP%]{margin-left:20%;margin-right:20%}"]
                        })
                    }
                    return n
                })(),
                rs = (() => {
                    class n {
                        static \u0275fac = function(o) {
                            return new(o || n)
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-privacy-security"]
                            ],
                            decls: 1,
                            vars: 0,
                            template: function(o, i) {
                                1 & o && t.nrm(0, "router-outlet")
                            },
                            dependencies: [R.n3],
                            encapsulation: 2
                        })
                    }
                    return n
                })();

            function ss(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "div", 5)(1, "p", 7), t.EFF(2, "2FA_SUCCESSFUL_SETUP"), t.k0s(), t.j41(3, "form", 8), t.bIt("ngSubmit", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.disable())
                    }), t.j41(4, "h2", 9), t.EFF(5, "REMOVE_TWO_FACTOR_AUTH"), t.k0s(), t.j41(6, "div", 10), t.EFF(7, "2FA_SETUP_ERROR"), t.k0s(), t.j41(8, "mat-form-field", 11)(9, "mat-label", 7), t.EFF(10, "LABEL_CURRENT_PASSWORD"), t.k0s(), t.nrm(11, "input", 12), t.nI1(12, "translate"), t.k0s(), t.j41(13, "button", 13)(14, "mat-icon"), t.EFF(15, "remove_selection"), t.k0s(), t.EFF(16), t.nI1(17, "translate"), t.k0s()()()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.R7$(3), t.Y8G("formGroup", e.twoFactorDisableForm), t.R7$(3), t.Y8G("hidden", !(e.errored && !e.twoFactorDisableForm.dirty)), t.R7$(5), t.Y8G("placeholder", t.mNQ(t.bMT(12, 6, "MANDATORY_CURRENT_PASSWORD"))), t.R7$(2), t.Y8G("disabled", e.twoFactorDisableForm.invalid), t.R7$(3), t.SpI(" ", t.bMT(17, 8, "BTN_REMOVE"), " ")
                }
            }

            function ls(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "form", 14), t.bIt("ngSubmit", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.setup())
                    }), t.j41(1, "span", 15), t.EFF(2, "2FA_AUTH_SETUP_INSTRUCTIONS"), t.k0s(), t.j41(3, "div", 16), t.nrm(4, "qr-code", 17), t.k0s(), t.j41(5, "div", 10), t.EFF(6, "2FA_SETUP_ERROR"), t.k0s(), t.j41(7, "mat-form-field", 11)(8, "mat-label", 7), t.EFF(9, "LABEL_CURRENT_PASSWORD"), t.k0s(), t.nrm(10, "input", 18), t.nI1(11, "translate"), t.k0s(), t.j41(12, "mat-form-field", 11)(13, "mat-label", 7), t.EFF(14, "INITIAL_CODE"), t.k0s(), t.nrm(15, "input", 19, 0), t.nI1(17, "translate"), t.j41(18, "mat-error", 7), t.EFF(19, "INVALID_TWO_FACTOR_AUTH_TOKEN"), t.k0s(), t.j41(20, "mat-icon", 20), t.nI1(21, "translate"), t.EFF(22, " help_outline "), t.k0s(), t.j41(23, "mat-hint", 21), t.EFF(24), t.k0s()(), t.j41(25, "button", 22)(26, "mat-icon"), t.EFF(27, "save"), t.k0s(), t.EFF(28), t.nI1(29, "translate"), t.k0s()()
                }
                if (2 & n) {
                    const e = t.sdS(16),
                        o = t.XpG();
                    t.Y8G("formGroup", o.twoFactorSetupForm), t.R7$(4), t.Y8G("value", o.totpUrl)("size", 300)("errorCorrectionLevel", "L"), t.R7$(), t.Y8G("hidden", !(o.errored && !o.twoFactorSetupForm.dirty)), t.R7$(5), t.Y8G("placeholder", t.mNQ(t.bMT(11, 15, "MANDATORY_CURRENT_PASSWORD"))), t.R7$(5), t.Y8G("placeholder", t.mNQ(t.bMT(17, 17, "INITIAL_CODE_PLACEHOLDER"))), t.BMQ("data-test-totp-secret", o.totpSecret), t.R7$(5), t.Y8G("matTooltip", t.mNQ(t.bMT(21, 19, "INITIAL_TOKEN_TOOLTIP"))), t.R7$(4), t.SpI("", (null == e.value ? null : e.value.length) || 0, "/6"), t.R7$(), t.Y8G("disabled", o.twoFactorSetupForm.invalid), t.R7$(3), t.SpI(" ", t.bMT(29, 21, "BTN_SAVE"), " ")
                }
            }
            S.Yv.add(C.HEq, C.hSh);
            let cs = (() => {
                    class n {
                        twoFactorAuthService;
                        configurationService;
                        snackBar;
                        translateService;
                        snackBarHelperService;
                        data;
                        twoFactorSetupForm = new l.J3({
                            passwordControl: new l.hs("", [l.k0.required]),
                            initialTokenControl: new l.hs("", [l.k0.required, l.k0.pattern("^[\\d]{6}$")])
                        });
                        twoFactorDisableForm = new l.J3({
                            passwordControl: new l.hs("", [l.k0.required])
                        });
                        setupStatus = null;
                        errored = null;
                        totpUrl;
                        totpSecret;
                        setupToken;
                        appName = "OWASP Juice Shop";
                        constructor(e, o, i, r, c) {
                            this.twoFactorAuthService = e, this.configurationService = o, this.snackBar = i, this.translateService = r, this.snackBarHelperService = c
                        }
                        ngOnInit() {
                            this.updateStatus()
                        }
                        updateStatus() {
                            const e = this.twoFactorAuthService.status(),
                                o = this.configurationService.getApplicationConfiguration();
                            return (0, wt.p)([e, o]).subscribe({
                                next: ([{
                                    setup: i,
                                    email: r,
                                    secret: c,
                                    setupToken: p
                                }, k]) => {
                                    if (this.setupStatus = i, this.appName = k.application.name, !i) {
                                        const x = encodeURIComponent(this.appName);
                                        this.totpUrl = `otpauth://totp/${x}:${r}?secret=${c}&issuer=${x}`, this.totpSecret = c, this.setupToken = p
                                    }
                                },
                                error: () => {
                                    console.log("Failed to fetch 2fa status")
                                }
                            }), e
                        }
                        setup() {
                            this.twoFactorAuthService.setup(this.twoFactorSetupForm.get("passwordControl")?.value, this.twoFactorSetupForm.get("initialTokenControl")?.value, this.setupToken).subscribe({
                                next: () => {
                                    this.setupStatus = !0, this.snackBarHelperService.open("CONFIRM_2FA_SETUP")
                                },
                                error: () => {
                                    this.twoFactorSetupForm.get("passwordControl")?.markAsPristine(), this.twoFactorSetupForm.get("initialTokenControl")?.markAsPristine(), this.errored = !0
                                }
                            })
                        }
                        disable() {
                            this.twoFactorAuthService.disable(this.twoFactorDisableForm.get("passwordControl")?.value).subscribe({
                                next: () => {
                                    this.updateStatus().subscribe(() => {
                                        this.setupStatus = !1
                                    }), this.snackBarHelperService.open("CONFIRM_2FA_DISABLE")
                                },
                                error: () => {
                                    this.twoFactorDisableForm.get("passwordControl")?.markAsPristine(), this.errored = !0
                                }
                            })
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(Te), t.rXU(M.s), t.rXU(rt.UG), t.rXU(d.c$), t.rXU(L.r))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-two-factor-auth"]
                            ],
                            decls: 7,
                            vars: 1,
                            consts: [
                                ["initToken", ""],
                                [1, "center-container"],
                                ["appearance", "outlined", 1, "mat-elevation-z6"],
                                [1, "mdc-card"],
                                ["id", "2fa-setup-title", "translate", ""],
                                ["id", "2fa-setup-successfully"],
                                ["id", "two-factor-auth-setup", 1, "form-container", 3, "formGroup"],
                                ["translate", ""],
                                ["id", "two-factor-auth-disable", 1, "form-container", 3, "ngSubmit", "formGroup"],
                                ["id", "two-factor-auth-disable-title", "translate", ""],
                                ["translate", "", 1, "error", 3, "hidden"],
                                ["appearance", "outline", "color", "accent"],
                                ["id", "currentPasswordDisable", "formControlName", "passwordControl", "type", "password", "matInput", "", "aria-label", "Field to enter the current password", "autocomplete", "off", 3, "placeholder"],
                                ["type", "submit", "id", "disableTwoFactorAuth", "mat-raised-button", "", "color", "warn", "aria-label", "Button to remove the two-factor authentication", 3, "disabled"],
                                ["id", "two-factor-auth-setup", 1, "form-container", 3, "ngSubmit", "formGroup"],
                                ["id", "2fa-setup-instructions", "translate", ""],
                                [1, "two-factor-qr-code", "center-container"],
                                [3, "value", "size", "errorCorrectionLevel"],
                                ["id", "currentPasswordSetup", "formControlName", "passwordControl", "type", "password", "matInput", "", "aria-label", "Field to enter the current password", "autocomplete", "off", 3, "placeholder"],
                                ["id", "initialToken", "formControlName", "initialTokenControl", "type", "text", "matInput", "", "aria-label", "Field to enter the initial token. This must have 6 digits.", "minlength", "6", "maxlength", "6", "pattern", "^[\\d]{6}$", 3, "placeholder"],
                                ["matSuffix", "", "matTooltipPosition", "right", "aria-label", "The token to be entered from the authenticator must have 6 digits.", 3, "matTooltip"],
                                ["align", "end"],
                                ["type", "submit", "id", "setupTwoFactorAuth", "mat-raised-button", "", "color", "primary", "aria-label", "Button to complete the two-factor configuration", 3, "disabled"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "div", 1)(1, "mat-card", 2)(2, "div", 3)(3, "h1", 4), t.EFF(4, "TITLE_TWO_FACTOR_AUTH_CONFIG"), t.k0s(), t.nVh(5, ss, 18, 10, "div", 5)(6, ls, 30, 23, "form", 6), t.k0s()()()), 2 & o && (t.R7$(5), t.vxM(!0 === i.setupStatus ? 5 : 6))
                            },
                            dependencies: [h.Hu, h.RN, d.h, d.Mm, l.YN, l.qT, l.me, l.BC, l.cb, l.xh, l.tU, l.R_, l.X1, l.j4, l.JD, I.R, _.rl, _.nJ, _.MV, _.TL, _.yw, T.fS, T.fg, b.Hl, b.$z, pt.XK, pt.Um, f.m_, f.An, B.oV, d.D9],
                            styles: ["\n\n\n\nmat-card[_ngcontent-%COMP%]{margin-bottom:50px;margin-left:auto;margin-right:auto;min-width:320px;width:30%}mat-form-field[_ngcontent-%COMP%]{padding-top:10px}.form-container[_ngcontent-%COMP%]{display:flex;flex-direction:column;position:relative}#setupTwoFactorAuth[_ngcontent-%COMP%], #disableTwoFactorAuth[_ngcontent-%COMP%]{margin-left:20%;margin-top:30px;width:60%}.two-factor-qr-code[_ngcontent-%COMP%]{margin:16px 0}#two-factor-auth-disable[_ngcontent-%COMP%]{margin-top:32px}#two-factor-auth-disable-title[_ngcontent-%COMP%]{margin-bottom:0}button[_ngcontent-%COMP%]{margin-top:5px}.center-container[_ngcontent-%COMP%]{align-items:stretch;display:flex;justify-content:center}"]
                        })
                    }
                    return n
                })(),
                ye = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        constructor(e) {
                            this.http = e
                        }
                        getCaptcha() {
                            return this.http.get(this.hostServer + "/rest/image-captcha/").pipe((0, g.W)(e => {
                                throw e
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })(),
                Re = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/rest/user";
                        constructor(e) {
                            this.http = e
                        }
                        erase(e) {
                            return this.http.post(this.host + "/erasure-request", e).pipe((0, g.W)(o => {
                                throw o
                            }))
                        }
                        dataExport(e) {
                            return this.http.post(this.host + "/data-export", e).pipe((0, g.W)(o => {
                                throw o
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })();
            const ds = () => ({
                length: "5"
            });

            function ms(n, a) {
                if (1 & n && (t.j41(0, "div")(1, "p", 13), t.EFF(2), t.k0s()()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(2), t.JRh(e.error)
                }
            }

            function ps(n, a) {
                1 & n && (t.j41(0, "mat-error", 4), t.EFF(1, "MANDATORY_CAPTCHA "), t.k0s())
            }

            function us(n, a) {
                1 & n && (t.j41(0, "mat-error", 4), t.EFF(1, "MANDATORY_CAPTCHA "), t.k0s())
            }

            function hs(n, a) {
                if (1 & n && (t.j41(0, "div")(1, "div")(2, "span", 14), t.EFF(3, "CAPTCHA "), t.j41(4, "span"), t.EFF(5, ":"), t.k0s()(), t.nrm(6, "div", 15), t.k0s(), t.j41(7, "mat-form-field", 16)(8, "mat-label", 4), t.EFF(9, "ENTER_CAPTCHA"), t.k0s(), t.nrm(10, "input", 17, 0), t.nI1(12, "translate"), t.j41(13, "mat-hint", 18), t.EFF(14), t.k0s(), t.nVh(15, ps, 2, 0, "mat-error", 4), t.nVh(16, us, 2, 0, "mat-error", 4), t.k0s()()), 2 & n) {
                    const e = t.sdS(11),
                        o = t.XpG();
                    t.R7$(6), t.Y8G("innerHTML", o.captcha, t.npT), t.R7$(4), t.Y8G("placeholder", t.mNQ(t.i5U(12, 7, "TYPE_THESE_LETTERS", t.lJ4(10, ds))))("formControl", o.captchaControl), t.R7$(4), t.SpI("", (null == e.value ? null : e.value.length) || 0, "/5"), t.R7$(), t.vxM(o.captchaControl.invalid && o.captchaControl.errors.required ? 15 : -1), t.R7$(), t.vxM(o.captchaControl.invalid && o.captchaControl.errors.minlength ? 16 : -1)
                }
            }
            let gs = (() => {
                    class n {
                        sanitizer;
                        imageCaptchaService;
                        dataSubjectService;
                        captchaControl = new l.hs("", [l.k0.required, l.k0.minLength(5)]);
                        formatControl = new l.hs("", [l.k0.required]);
                        captcha;
                        dataRequest = void 0;
                        confirmation;
                        error;
                        lastSuccessfulTry;
                        presenceOfCaptcha = !1;
                        userData;
                        constructor(e, o, i) {
                            this.sanitizer = e, this.imageCaptchaService = o, this.dataSubjectService = i
                        }
                        ngOnInit() {
                            this.needCaptcha(), this.dataRequest = {}
                        }
                        needCaptcha() {
                            const e = new Date,
                                o = localStorage.getItem("lstdtxprt") ? new Date(JSON.parse(String(localStorage.getItem("lstdtxprt")))) : new Date(0);
                            e.getTime() - o.getTime() < 3e5 && (this.getNewCaptcha(), this.presenceOfCaptcha = !0)
                        }
                        getNewCaptcha() {
                            this.imageCaptchaService.getCaptcha().subscribe(e => {
                                this.captcha = this.sanitizer.bypassSecurityTrustHtml(e.image)
                            })
                        }
                        save() {
                            this.presenceOfCaptcha && (this.dataRequest.answer = this.captchaControl.value), this.dataRequest.format = this.formatControl.value, this.dataSubjectService.dataExport(this.dataRequest).subscribe({
                                next: e => {
                                    this.error = null, this.confirmation = e.confirmation, this.userData = e.userData, window.open("", "_blank", "width=500")?.document.write(this.userData), this.lastSuccessfulTry = new Date, localStorage.setItem("lstdtxprt", JSON.stringify(this.lastSuccessfulTry)), this.ngOnInit(), this.resetForm()
                                },
                                error: e => {
                                    this.error = e.error, this.confirmation = null, this.resetFormError()
                                }
                            })
                        }
                        resetForm() {
                            this.captchaControl.markAsUntouched(), this.captchaControl.markAsPristine(), this.captchaControl.setValue(""), this.formatControl.markAsUntouched(), this.formatControl.markAsPristine(), this.formatControl.setValue("")
                        }
                        resetFormError() {
                            this.captchaControl.markAsUntouched(), this.captchaControl.markAsPristine(), this.captchaControl.setValue("")
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(it.up), t.rXU(ye), t.rXU(Re))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-data-export"]
                            ],
                            decls: 27,
                            vars: 7,
                            consts: [
                                ["captchaInput", ""],
                                [1, "center-container"],
                                ["appearance", "outlined", 1, "mat-elevation-z6"],
                                [1, "mdc-card"],
                                ["translate", ""],
                                ["id", "data-export-form", 1, "form-container"],
                                ["id", "formatControl", "aria-label", "Select an option", 3, "formControl"],
                                ["translate", "", 1, "radio-label"],
                                ["value", "1", "aria-label", "Export Option JSON"],
                                ["value", "2", "aria-label", "Export Option PDF", "disabled", ""],
                                ["value", "3", "aria-label", "Export Option Excel", "disabled", ""],
                                ["type", "submit", "id", "submitButton", "color", "primary", "mat-raised-button", "", "aria-label", "Button to send the request", 2, "margin-top", "15px", 3, "click", "disabled"],
                                [1, "hint"],
                                [1, "error"],
                                ["translate", "", 2, "float", "left", "margin-top", "40px", "font-weight", "500", "margin-bottom", "20px"],
                                [1, "captcha-image", 2, "margin-left", "10px", 3, "innerHTML"],
                                ["appearance", "outline", "color", "accent", 2, "margin-top", "10px", "width", "100%"],
                                ["type", "text", "matInput", "", "maxlength", "5", "aria-label", "Input for the CAPTCHA", 3, "formControl", "placeholder"],
                                ["align", "end"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "div", 1)(1, "mat-card", 2)(2, "div", 3)(3, "h1", 4), t.EFF(4, "TITLE_REQUEST_DATA_EXPORT"), t.k0s(), t.nVh(5, ms, 3, 1, "div"), t.j41(6, "div", 5)(7, "mat-radio-group", 6)(8, "mat-label", 7), t.EFF(9, "EXPORT_LABEL "), t.j41(10, "span"), t.EFF(11, ":"), t.k0s()(), t.j41(12, "mat-radio-button", 8), t.EFF(13, "JSON"), t.k0s(), t.j41(14, "mat-radio-button", 9), t.EFF(15, "PDF"), t.k0s(), t.j41(16, "mat-radio-button", 10), t.EFF(17, "Excel"), t.k0s()(), t.nVh(18, hs, 17, 11, "div"), t.k0s(), t.j41(19, "button", 11), t.bIt("click", function() {
                                    return i.save()
                                }), t.j41(20, "mat-icon"), t.EFF(21, "download"), t.k0s(), t.EFF(22), t.nI1(23, "translate"), t.k0s(), t.j41(24, "div", 12)(25, "span", 4), t.EFF(26, "DATA_EXPORT_HINT"), t.k0s()()()()()), 2 & o && (t.R7$(5), t.vxM(i.error ? 5 : -1), t.R7$(2), t.Y8G("formControl", i.formatControl), t.R7$(11), t.vxM(i.presenceOfCaptcha ? 18 : -1), t.R7$(), t.Y8G("disabled", i.formatControl.invalid || i.captchaControl.invalid && i.presenceOfCaptcha), t.R7$(3), t.SpI(" ", t.bMT(23, 5, "BTN_REQUEST"), " "))
                            },
                            dependencies: [h.Hu, h.RN, d.h, d.Mm, ot.VT, l.YN, l.me, l.BC, l.tU, l.X1, l.l_, _.nJ, ot._g, I.R, _.rl, _.MV, _.TL, T.fS, T.fg, b.Hl, b.$z, f.m_, f.An, d.D9],
                            styles: ["\n\n\n\nmat-card[_ngcontent-%COMP%]{height:auto;margin-bottom:50px;margin-left:auto;margin-right:auto;min-width:320px;width:25%}#submitButton[_ngcontent-%COMP%]{margin-left:20%;width:60%}.hint[_ngcontent-%COMP%]{display:flex;font-size:10px;justify-content:center;margin-top:5px}.mat-mdc-radio-button[_ngcontent-%COMP%] ~ .mat-mdc-radio-button[_ngcontent-%COMP%]{margin-bottom:16px;margin-left:16px}.radio-label[_ngcontent-%COMP%]{font-weight:500;margin-right:8px}.captcha-image[_ngcontent-%COMP%]{float:left;margin-bottom:10px;margin-top:20px}.center-container[_ngcontent-%COMP%]{align-items:stretch;display:flex;justify-content:center}"]
                        })
                    }
                    return n
                })(),
                fs = (() => {
                    class n {
                        sanitizer;
                        lastLoginIp = "?";
                        constructor(e) {
                            this.sanitizer = e
                        }
                        ngOnInit() {
                            try {
                                this.parseAuthToken()
                            } catch (e) {
                                console.log(e)
                            }
                        }
                        parseAuthToken() {
                            let e = {};
                            const o = localStorage.getItem("token");
                            o && (e = oe(o), e.data.lastLoginIp && (this.lastLoginIp = this.sanitizer.bypassSecurityTrustHtml(`<small>${e.data.lastLoginIp}</small>`)))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(it.up))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-last-login-ip"]
                            ],
                            decls: 9,
                            vars: 1,
                            consts: [
                                ["appearance", "outlined", 1, "ipCard", "mat-elevation-z6"],
                                [1, "mdc-card"],
                                ["translate", ""],
                                [2, "margin-bottom", "15px"],
                                [3, "innerHTML"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "mat-card", 0)(1, "div", 1)(2, "h1", 2), t.EFF(3, "LAST_LOGIN_IP"), t.k0s(), t.j41(4, "p", 3)(5, "span", 2), t.EFF(6, "IP_ADDRESS"), t.k0s(), t.EFF(7, "\xa0\xa0"), t.nrm(8, "span", 4), t.k0s()()()), 2 & o && (t.R7$(8), t.Y8G("innerHTML", i.lastLoginIp, t.npT))
                            },
                            dependencies: [h.Hu, h.RN, d.h, d.Mm],
                            styles: ["\n\n\n\n.ipCard[_ngcontent-%COMP%]{display:block;margin:0 auto 5%;min-width:320px;width:25%}"]
                        })
                    }
                    return n
                })(),
                _s = (() => {
                    class n {
                        _document;
                        configurationService;
                        applicationName = "OWASP Juice Shop";
                        privacyContactEmail;
                        applicationUrl;
                        constructor(e, o) {
                            this._document = e, this.configurationService = o
                        }
                        ngOnInit() {
                            this.applicationUrl = this._document.location.protocol + "//" + this._document.location.hostname, this.configurationService.getApplicationConfiguration().subscribe({
                                next: e => {
                                    e?.application?.name && (this.applicationName = e.application.name), this.privacyContactEmail = e?.application?.privacyContactEmail ? e.application.privacyContactEmail : `donotreply@${this._document.location.hostname}`
                                },
                                error: e => {
                                    console.log(e)
                                }
                            })
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(s.qQL), t.rXU(M.s))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-privacy-policy"]
                            ],
                            decls: 171,
                            vars: 12,
                            consts: [
                                ["appearance", "outlined", 1, "mat-elevation-z6"],
                                [1, "mdc-card"],
                                [1, "privacy-policy"],
                                [1, "hot"],
                                ["href", "https://www.freeprivacypolicy.com/", "aria-label", "Link to the page where the privacy policy was created"],
                                [1, "divider"],
                                ["src", "assets/public/images/padding/81px.png"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "mat-card", 0)(1, "div", 1)(2, "section", 2)(3, "h1"), t.EFF(4), t.nI1(5, "translate"), t.k0s(), t.j41(6, "p"), t.EFF(7, "Effective date: March 15, 2019"), t.k0s(), t.j41(8, "p"), t.EFF(9), t.j41(10, "span", 3), t.EFF(11), t.k0s(), t.EFF(12, ' website (the "Service").'), t.k0s(), t.j41(13, "p"), t.EFF(14), t.j41(15, "a", 4), t.EFF(16, "Free Privacy Policy website"), t.k0s(), t.EFF(17, "."), t.k0s(), t.j41(18, "p"), t.EFF(19), t.k0s(), t.nrm(20, "mat-divider", 5), t.j41(21, "h2"), t.EFF(22, "A. Information Collection And Use"), t.k0s(), t.j41(23, "p"), t.EFF(24, "We collect several different types of information for various purposes to provide and improve our Service to you."), t.k0s(), t.j41(25, "h3"), t.EFF(26, "A1. Types of Data Collected"), t.k0s(), t.j41(27, "h4"), t.EFF(28, "A1.1 Personal Data"), t.k0s(), t.j41(29, "p"), t.EFF(30, 'While using our Service, we may ask you to provide us with certain personally identifiable information that can be used to contact or identify you ("Personal Data"). Personally identifiable information may include, but is not limited to:'), t.k0s(), t.j41(31, "ul")(32, "li"), t.EFF(33, "Email address"), t.k0s(), t.j41(34, "li"), t.EFF(35, "Address, State, Province, ZIP/Postal code, City"), t.k0s(), t.j41(36, "li"), t.EFF(37, "Cookies and Usage Data"), t.k0s()(), t.j41(38, "h4"), t.EFF(39, "A1.2 Usage Data"), t.k0s(), t.j41(40, "p")(41, "span", 3), t.EFF(42, "We may also"), t.k0s(), t.EFF(43, ' collect information how the Service is accessed and used ("Usage Data"). This Usage Data may include information such as your computer\'s Internet Protocol address (e.g. IP address), browser type, browser version, the pages of our Service that you visit, the time and date of your visit, the time spent on those pages, unique device identifiers and other diagnostic data.'), t.k0s(), t.j41(44, "h4"), t.EFF(45, "A1.3 Tracking & Cookies Data"), t.k0s(), t.j41(46, "p"), t.EFF(47, "We use cookies and similar tracking technologies to track the activity on our Service and hold certain information."), t.k0s(), t.j41(48, "p"), t.EFF(49, "Cookies are files with small amount of data which may include an anonymous unique identifier. Cookies are sent to your browser from a website and stored on your device. Tracking technologies also used are beacons, tags, and scripts to collect and track information and to improve and analyze our Service."), t.k0s(), t.j41(50, "p"), t.EFF(51, "You can "), t.j41(52, "span", 3), t.EFF(53, "instruct you"), t.k0s(), t.EFF(54, "r browser "), t.j41(55, "span", 3), t.EFF(56, "to refuse all"), t.k0s(), t.EFF(57, " cookies or to indicate when a cookie is being sent. However, if you do not accept cookies, you may not be able to use some portions of our Service."), t.k0s(), t.j41(58, "p"), t.EFF(59, "Examples of Cookies we use:"), t.k0s(), t.j41(60, "ul")(61, "li")(62, "strong"), t.EFF(63, "Session Cookies:"), t.k0s(), t.EFF(64, " \xa0We use Session Cookies to operate our Service."), t.k0s(), t.j41(65, "li")(66, "strong"), t.EFF(67, "Preference Cookies:"), t.k0s(), t.EFF(68, "\xa0We use Preference Cookies to remember your preferences and various settings. "), t.k0s(), t.j41(69, "li")(70, "strong"), t.EFF(71, "Security Cookies:"), t.k0s(), t.EFF(72, "\xa0We use Security Cookies for security purposes."), t.k0s()(), t.j41(73, "h3"), t.EFF(74, "A2. Use of Data"), t.k0s(), t.j41(75, "p"), t.EFF(76), t.k0s(), t.j41(77, "ul")(78, "li"), t.EFF(79, "To provide and maintain the Service"), t.k0s(), t.j41(80, "li"), t.EFF(81, "To notify you about changes to our Service"), t.k0s(), t.j41(82, "li"), t.EFF(83, "To allow you to participate in interactive features of our Service when you choose to do so"), t.k0s(), t.j41(84, "li"), t.EFF(85, "To provide customer care and support"), t.k0s(), t.j41(86, "li"), t.EFF(87, "To provide analysis or valuable information so that we can improve the Service"), t.k0s(), t.j41(88, "li"), t.EFF(89, "To monitor the usage of the Service"), t.k0s(), t.j41(90, "li"), t.EFF(91, "To detect, prevent and address technical issues"), t.k0s()(), t.nrm(92, "mat-divider", 5), t.j41(93, "h2"), t.EFF(94, "B. Transfer Of Data"), t.k0s(), t.j41(95, "p"), t.EFF(96, "Your information, including Personal Data, may be transferred to \u2014 and maintained on \u2014 computers located outside of your state, province, country or other governmental jurisdiction where the data protection laws may differ than those from your jurisdiction."), t.k0s(), t.j41(97, "p"), t.EFF(98, "If you are located outside Germany and choose to provide information to us, please note that we transfer the data, including Personal Data, to Germany and process it there."), t.k0s(), t.j41(99, "p"), t.EFF(100, "Your consent to this Privacy Policy followed by your submission of such information represents your agreement to that transfer."), t.k0s(), t.j41(101, "p"), t.EFF(102), t.j41(103, "span", 3), t.EFF(104, "reasonably necessary"), t.k0s(), t.EFF(105, " to ensure that your data is treated securely and in accordance with this Privacy Policy and no transfer of your Personal Data will take place to an organization or a country unless there are adequate controls in place including the security of your data and other personal information."), t.k0s(), t.nrm(106, "mat-divider", 5), t.j41(107, "h2"), t.EFF(108, "C. Disclosure Of Data"), t.k0s(), t.j41(109, "h3"), t.EFF(110, "C1. Legal Requirements"), t.k0s(), t.j41(111, "p"), t.EFF(112), t.k0s(), t.j41(113, "ul")(114, "li"), t.EFF(115, "To comply with a legal obligation"), t.k0s(), t.j41(116, "li"), t.EFF(117), t.k0s(), t.j41(118, "li"), t.EFF(119, "To prevent or investigate possible wrongdoing in connection with the Service"), t.k0s(), t.j41(120, "li"), t.EFF(121, "To protect the personal safety of users of the Service or the public"), t.k0s(), t.j41(122, "li"), t.EFF(123, "To protect against legal liability"), t.k0s()(), t.nrm(124, "mat-divider", 5), t.j41(125, "h2"), t.EFF(126, "D. Security Of Data"), t.k0s(), t.j41(127, "p"), t.EFF(128, "The security of your data is important to us, but remember that no method of transmission over the Internet, or method of electronic storage is 100% secure. While we strive to use commercially acceptable means to protect your Personal Data, we cannot guarantee its absolute security."), t.k0s(), t.nrm(129, "mat-divider", 5), t.j41(130, "h2"), t.EFF(131, "E. Service Providers"), t.k0s(), t.j41(132, "p"), t.EFF(133, 'We may employ third party companies and individuals to facilitate our Service ("Service Providers"), to provide the Service on our behalf, to perform Service-related services or to assist us in analyzing how our Service is used.'), t.k0s(), t.j41(134, "p"), t.EFF(135, "These third parties have access to your Personal Data only to perform these tasks on our behalf and are obligated not to disclose or use it for any other purpose."), t.k0s(), t.nrm(136, "mat-divider", 5), t.j41(137, "h2"), t.EFF(138, "F. Links To Other Sites"), t.k0s(), t.j41(139, "p"), t.EFF(140, "Our Service may contain links to other sites that are not operated by us. If you click on a third party link, you will be directed to that third party's site. We strongly advise you to review the Privacy Policy of every site you visit."), t.k0s(), t.j41(141, "p"), t.EFF(142, "We have no control over and assume no "), t.j41(143, "span", 3), t.EFF(144, "responsibility"), t.k0s(), t.EFF(145, " for the content, privacy policies or practices of any third party sites or services."), t.k0s(), t.nrm(146, "mat-divider", 5), t.j41(147, "h2"), t.EFF(148, "G. Children's Privacy"), t.k0s(), t.j41(149, "p"), t.EFF(150, 'Our Service does not address anyone under the age of 18 ("Children").'), t.k0s(), t.j41(151, "p"), t.EFF(152, "We do not knowingly collect personally identifiable information from anyone under the age of 18. If you are a parent or guardian and you are aware that your Children has provided us with Personal Data, please contact us. If we become aware that we have collected Personal Data from children without verification of parental consent, we take steps to remove that information from our servers."), t.k0s(), t.nrm(153, "mat-divider", 5), t.j41(154, "h2"), t.EFF(155, "H. Changes To This Privacy Policy"), t.k0s(), t.j41(156, "p"), t.EFF(157, "We may update our Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy Policy on this page."), t.k0s(), t.j41(158, "p"), t.EFF(159, 'We will let you know via email and/or a prominent notice on our Service, prior to the change becoming effective and update the "effective date" at the top of this Privacy Policy.'), t.k0s(), t.j41(160, "p"), t.EFF(161, "You are advised to review this Privacy Policy periodically for any changes. Changes to this Privacy Policy are effective when they are posted on this page."), t.k0s(), t.nrm(162, "mat-divider", 5), t.j41(163, "h2"), t.EFF(164, "Contact Us"), t.k0s(), t.j41(165, "p"), t.EFF(166, "If you have any questions about this Privacy Policy, please contact us:"), t.k0s(), t.j41(167, "ul")(168, "li"), t.EFF(169), t.k0s()()(), t.nrm(170, "img", 6), t.k0s()()), 2 & o && (t.R7$(4), t.JRh(t.bMT(5, 10, "TITLE_PRIVACY_POLICY")), t.R7$(5), t.SpI("", i.applicationName, ' ("us", "we", or "our") operates the '), t.R7$(2), t.JRh(i.applicationUrl), t.R7$(3), t.SpI("This page informs you of our policies regarding the collection, use, and disclosure of personal data when you use our Service and the choices you have associated with that data. Our Privacy Policy for ", i.applicationName, " is created with the help of the "), t.R7$(5), t.SpI("We use your data to provide and improve the Service. By using the Service, you agree to the collection and use of information in accordance with this policy. Unless otherwise defined in this Privacy Policy, terms used in this Privacy Policy have the same meanings as in our Terms and Conditions, accessible from ", i.applicationUrl, " ."), t.R7$(57), t.SpI("", i.applicationName, " uses the collected data for various purposes:"), t.R7$(26), t.SpI("", i.applicationName, " will take all steps "), t.R7$(10), t.SpI("", i.applicationName, " may disclose your Personal Data in the good faith belief that such action is necessary to:"), t.R7$(5), t.SpI("To protect and defend the rights or property of ", i.applicationName), t.R7$(52), t.SpI("By email: ", i.privacyContactEmail))
                            },
                            dependencies: [h.Hu, h.RN, d.h, U.q, d.D9],
                            styles: ["\n\n\n\n.privacy-policy[_ngcontent-%COMP%]{margin-bottom:15px;margin-left:8.33333%;width:83.3333%}p[_ngcontent-%COMP%]{margin-bottom:15px;text-align:justify;text-justify:inter-word}mat-card[_ngcontent-%COMP%]{margin-bottom:50px;margin-left:auto;margin-right:auto;width:90%}h2[_ngcontent-%COMP%]{padding-top:10px}h3[_ngcontent-%COMP%]{font-weight:550}strong[_ngcontent-%COMP%]{font-style:italic;font-weight:550}h4[_ngcontent-%COMP%]{text-decoration:underline}.hot[_ngcontent-%COMP%]:hover{text-shadow:0 0 20px #fefcc9,10px -10px 30px #feec85,-20px -20px 40px #ffae34,20px -40px 50px #ec760c,-20px -60px 60px #cd4606,0 -80px 70px #973716,10px -90px 80px #451b0e}.divider[_ngcontent-%COMP%]{border-top-width:4px!important;margin-left:12%!important;width:82%!important}.mdc-card[_ngcontent-%COMP%]{border:0}.mdc-card[_ngcontent-%COMP%]   img[_ngcontent-%COMP%]{max-height:10px}"]
                        })
                    }
                    return n
                })();
            var zt = m(7303);
            const Cs = () => ({
                    length: "160"
                }),
                bs = () => ({
                    range: "1000000-9999999999"
                });

            function vs(n, a) {
                1 & n && (t.j41(0, "mat-error"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.SpI(" ", t.bMT(2, 1, "MANDATORY_COUNTRY"), " "))
            }

            function ks(n, a) {
                1 & n && (t.j41(0, "mat-error"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.SpI(" ", t.bMT(2, 1, "MANDATORY_NAME"), " "))
            }

            function Ss(n, a) {
                1 & n && (t.j41(0, "mat-error"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.SpI(" ", t.bMT(2, 1, "MANDATORY_NUMBER"), " "))
            }

            function Fs(n, a) {
                1 & n && (t.j41(0, "mat-error", 9), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.Y8G("translateParams", t.lJ4(4, bs)), t.R7$(), t.SpI(" ", t.bMT(2, 2, "INVALID_MOBILE_NUMBER"), " "))
            }

            function xs(n, a) {
                1 & n && (t.j41(0, "mat-error"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.SpI(" ", t.bMT(2, 1, "MANDATORY_ZIP"), " "))
            }

            function Ts(n, a) {
                1 & n && (t.j41(0, "mat-error"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.SpI(" ", t.bMT(2, 1, "MANDATORY_ADDRESS"), " "))
            }

            function ys(n, a) {
                1 & n && (t.j41(0, "mat-error"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.SpI(" ", t.bMT(2, 1, "MANDATORY_CITY"), " "))
            }
            let we = (() => {
                    class n {
                        location;
                        formSubmitService;
                        addressService;
                        router;
                        activatedRoute;
                        translate;
                        snackBarHelperService;
                        countryControl = new l.hs("", [l.k0.required]);
                        nameControl = new l.hs("", [l.k0.required]);
                        numberControl = new l.hs("", [l.k0.required, l.k0.min(1111111), l.k0.max(9999999999)]);
                        pinControl = new l.hs("", [l.k0.required, l.k0.maxLength(8)]);
                        addressControl = new l.hs("", [l.k0.required, l.k0.maxLength(160)]);
                        cityControl = new l.hs("", [l.k0.required]);
                        stateControl = new l.hs;
                        address = void 0;
                        mode = "create";
                        addressId = void 0;
                        constructor(e, o, i, r, c, p, k) {
                            this.location = e, this.formSubmitService = o, this.addressService = i, this.router = r, this.activatedRoute = c, this.translate = p, this.snackBarHelperService = k
                        }
                        ngOnInit() {
                            this.address = {}, this.activatedRoute.paramMap.subscribe(e => {
                                e.has("addressId") ? (this.mode = "edit", this.addressId = e.get("addressId"), this.addressService.getById(this.addressId).subscribe(o => {
                                    this.initializeForm(o)
                                })) : (this.mode = "create", this.addressId = null)
                            }), this.formSubmitService.attachEnterKeyHandler("address-form", "submitButton", () => {
                                this.save()
                            })
                        }
                        save() {
                            this.address.country = this.countryControl.value, this.address.fullName = this.nameControl.value, this.address.mobileNum = this.numberControl.value, this.address.zipCode = this.pinControl.value, this.address.streetAddress = this.addressControl.value, this.address.city = this.cityControl.value, this.address.state = this.stateControl.value, "edit" === this.mode ? this.addressService.put(this.addressId, this.address).subscribe({
                                next: e => {
                                    this.address = {}, this.ngOnInit(), this.resetForm(), this.routeToPreviousUrl(), this.translate.get("ADDRESS_UPDATED", {
                                        city: e.city
                                    }).subscribe({
                                        next: o => {
                                            this.snackBarHelperService.open(o, "confirmBar")
                                        },
                                        error: o => {
                                            this.snackBarHelperService.open(o, "confirmBar")
                                        }
                                    })
                                },
                                error: e => {
                                    this.snackBarHelperService.open(e.error?.error, "errorBar"), this.address = {}, this.resetForm()
                                }
                            }) : this.addressService.save(this.address).subscribe({
                                next: e => {
                                    this.address = {}, this.ngOnInit(), this.resetForm(), this.routeToPreviousUrl(), this.translate.get("ADDRESS_ADDED", {
                                        city: e.city
                                    }).subscribe({
                                        next: o => {
                                            this.snackBarHelperService.open(o, "confirmBar")
                                        },
                                        error: o => {
                                            this.snackBarHelperService.open(o, "confirmBar")
                                        }
                                    })
                                },
                                error: e => {
                                    this.snackBarHelperService.open(e.error?.error, "errorBar"), this.address = {}, this.resetForm()
                                }
                            })
                        }
                        initializeForm(e) {
                            this.countryControl.setValue(e.country), this.nameControl.setValue(e.fullName), this.numberControl.setValue(e.mobileNum), this.pinControl.setValue(e.zipCode), this.addressControl.setValue(e.streetAddress), this.cityControl.setValue(e.city), this.stateControl.setValue(e.state)
                        }
                        routeToPreviousUrl() {
                            this.location.back()
                        }
                        resetForm() {
                            this.countryControl.markAsUntouched(), this.countryControl.markAsPristine(), this.countryControl.setValue(""), this.nameControl.markAsUntouched(), this.nameControl.markAsPristine(), this.nameControl.setValue(""), this.numberControl.markAsUntouched(), this.numberControl.markAsPristine(), this.numberControl.setValue(""), this.pinControl.markAsUntouched(), this.pinControl.markAsPristine(), this.pinControl.setValue(""), this.addressControl.markAsUntouched(), this.addressControl.markAsPristine(), this.addressControl.setValue(""), this.cityControl.markAsUntouched(), this.cityControl.markAsPristine(), this.cityControl.setValue(""), this.stateControl.markAsUntouched(), this.stateControl.markAsPristine(), this.stateControl.setValue("")
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(zt.aZ), t.rXU(tt), t.rXU(st), t.rXU(R.Ix), t.rXU(R.nX), t.rXU(d.c$), t.rXU(L.r))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-address-create"]
                            ],
                            decls: 78,
                            vars: 73,
                            consts: [
                                ["pin", ""],
                                ["adress", ""],
                                [1, "page-center"],
                                ["appearance", "outlined", 1, "mat-elevation-z6"],
                                [1, "mdc-card"],
                                ["id", "address-form", 1, "form-container"],
                                ["appearance", "outline", "color", "accent"],
                                ["type", "text", "matInput", "", 3, "formControl", "placeholder"],
                                ["type", "number", "matInput", "", 3, "formControl", "placeholder"],
                                [3, "translateParams"],
                                ["align", "end"],
                                ["appearance", "outline", "color", "accent", 1, "textarea-field"],
                                ["id", "address", "matInput", "", "matAutosizeMinRows", "4", "matAutosizeMaxRows", "4", "matTextareaAutosize", "", "cols", "50", "maxlength", "160", 3, "formControl", "placeholder"],
                                [1, "fas", "fa-exclamation-circle"],
                                [2, "margin-left", "5px"],
                                [1, "actions"],
                                ["mat-stroked-button", "", 1, "btn-return", 3, "click"],
                                ["type", "submit", "id", "submitButton", "mat-raised-button", "", "color", "primary", 3, "click", "disabled"]
                            ],
                            template: function(o, i) {
                                if (1 & o) {
                                    const r = t.RV6();
                                    t.j41(0, "div", 2)(1, "mat-card", 3)(2, "div", 4)(3, "h1"), t.EFF(4), t.nI1(5, "translate"), t.k0s(), t.j41(6, "div", 5)(7, "mat-form-field", 6)(8, "mat-label"), t.EFF(9), t.nI1(10, "translate"), t.k0s(), t.nrm(11, "input", 7), t.nI1(12, "translate"), t.nVh(13, vs, 3, 3, "mat-error"), t.k0s(), t.j41(14, "mat-form-field", 6)(15, "mat-label"), t.EFF(16), t.nI1(17, "translate"), t.k0s(), t.nrm(18, "input", 7), t.nI1(19, "translate"), t.nVh(20, ks, 3, 3, "mat-error"), t.k0s(), t.j41(21, "mat-form-field", 6)(22, "mat-label"), t.EFF(23), t.nI1(24, "translate"), t.k0s(), t.nrm(25, "input", 8), t.nI1(26, "translate"), t.nVh(27, Ss, 3, 3, "mat-error"), t.nVh(28, Fs, 3, 5, "mat-error", 9), t.k0s(), t.j41(29, "mat-form-field", 6)(30, "mat-label"), t.EFF(31), t.nI1(32, "translate"), t.k0s(), t.nrm(33, "input", 7, 0), t.nI1(35, "translate"), t.j41(36, "mat-hint", 10), t.EFF(37), t.k0s(), t.nVh(38, xs, 3, 3, "mat-error"), t.k0s(), t.j41(39, "mat-form-field", 11)(40, "mat-label"), t.EFF(41), t.nI1(42, "translate"), t.k0s(), t.nrm(43, "textarea", 12, 1), t.nI1(45, "translate"), t.j41(46, "mat-hint"), t.nrm(47, "i", 13), t.j41(48, "em", 14), t.EFF(49), t.nI1(50, "translate"), t.k0s()(), t.j41(51, "mat-hint", 10), t.EFF(52), t.k0s(), t.nVh(53, Ts, 3, 3, "mat-error"), t.k0s(), t.j41(54, "mat-form-field", 6)(55, "mat-label"), t.EFF(56), t.nI1(57, "translate"), t.k0s(), t.nrm(58, "input", 7), t.nI1(59, "translate"), t.nVh(60, ys, 3, 3, "mat-error"), t.k0s(), t.j41(61, "mat-form-field", 6)(62, "mat-label"), t.EFF(63), t.nI1(64, "translate"), t.k0s(), t.nrm(65, "input", 7), t.nI1(66, "translate"), t.k0s()(), t.j41(67, "div", 15)(68, "button", 16), t.bIt("click", function() {
                                        return s.eBV(r), s.Njj(i.routeToPreviousUrl())
                                    }), t.j41(69, "mat-icon"), t.EFF(70, "navigate_before"), t.k0s(), t.EFF(71), t.nI1(72, "translate"), t.k0s(), t.j41(73, "button", 17), t.bIt("click", function() {
                                        return s.eBV(r), s.Njj(i.save())
                                    }), t.j41(74, "mat-icon"), t.EFF(75, "send"), t.k0s(), t.EFF(76), t.nI1(77, "translate"), t.k0s()()()()()
                                }
                                if (2 & o) {
                                    const r = t.sdS(34),
                                        c = t.sdS(44);
                                    t.R7$(4), t.JRh(t.bMT(5, 35, "ADD_NEW_ADDRESS")), t.R7$(5), t.JRh(t.bMT(10, 37, "LABEL_COUNTRY")), t.R7$(2), t.Y8G("formControl", i.countryControl)("placeholder", t.bMT(12, 39, "MANDATORY_COUNTRY")), t.R7$(2), t.vxM(i.countryControl.invalid && i.countryControl.errors.required ? 13 : -1), t.R7$(3), t.JRh(t.bMT(17, 41, "LABEL_NAME")), t.R7$(2), t.Y8G("formControl", i.nameControl)("placeholder", t.bMT(19, 43, "MANDATORY_NAME")), t.R7$(2), t.vxM(i.nameControl.invalid && i.nameControl.errors.required ? 20 : -1), t.R7$(3), t.JRh(t.bMT(24, 45, "LABEL_MOBILE_NUMBER")), t.R7$(2), t.Y8G("formControl", i.numberControl)("placeholder", t.bMT(26, 47, "MANDATORY_NUMBER")), t.R7$(2), t.vxM(i.numberControl.invalid && i.numberControl.errors.required ? 27 : -1), t.R7$(), t.vxM(i.numberControl.invalid && (i.numberControl.errors.min || i.numberControl.errors.max) ? 28 : -1), t.R7$(3), t.JRh(t.bMT(32, 49, "LABEL_ZIP_CODE")), t.R7$(2), t.Y8G("formControl", i.pinControl)("placeholder", t.bMT(35, 51, "MANDATORY_ZIP")), t.R7$(4), t.SpI("", (null == r.value ? null : r.value.length) || 0, "/8"), t.R7$(), t.vxM(i.pinControl.invalid && i.pinControl.errors.required ? 38 : -1), t.R7$(3), t.JRh(t.bMT(42, 53, "LABEL_ADDRESS")), t.R7$(2), t.Y8G("formControl", i.addressControl)("placeholder", t.bMT(45, 55, "MANDATORY_ADDRESS")), t.R7$(6), t.JRh(t.i5U(50, 57, "MAX_TEXTAREA_LENGTH", t.lJ4(72, Cs))), t.R7$(3), t.SpI("", (null == c.value ? null : c.value.length) || 0, "/160"), t.R7$(), t.vxM(i.addressControl.invalid && i.addressControl.errors.required ? 53 : -1), t.R7$(3), t.JRh(t.bMT(57, 60, "LABEL_CITY")), t.R7$(2), t.Y8G("formControl", i.cityControl)("placeholder", t.bMT(59, 62, "MANDATORY_CITY")), t.R7$(2), t.vxM(i.cityControl.invalid && i.cityControl.errors.required ? 60 : -1), t.R7$(3), t.JRh(t.bMT(64, 64, "LABEL_STATE")), t.R7$(2), t.Y8G("formControl", i.stateControl)("placeholder", t.bMT(66, 66, "STATE_PLACEHOLDER")), t.R7$(6), t.SpI(" ", t.bMT(72, 68, "LABEL_BACK"), " "), t.R7$(2), t.Y8G("disabled", i.countryControl.invalid || i.nameControl.invalid || i.numberControl.invalid || i.pinControl.invalid || i.addressControl.invalid || i.cityControl.invalid), t.R7$(3), t.SpI(" ", t.bMT(77, 70, "BTN_SUBMIT"), " ")
                                }
                            },
                            dependencies: [h.Hu, h.RN, d.h, I.R, _.rl, _.nJ, _.MV, _.TL, T.fS, T.fg, l.YN, l.me, l.Q0, l.BC, l.tU, l.X1, l.l_, b.Hl, b.$z, f.m_, f.An, d.D9],
                            styles: ['\n\n\n\n.page-center[_ngcontent-%COMP%]{display:flex;justify-content:center;padding:16px}mat-card[_ngcontent-%COMP%]{height:auto;min-width:320px;width:30%}#submitButton[_ngcontent-%COMP%]{float:right}.form-container[_ngcontent-%COMP%]{display:flex;flex-direction:column;position:relative}.form-container[_ngcontent-%COMP%]   mat-form-field[_ngcontent-%COMP%]{width:100%}.btn-return[_ngcontent-%COMP%]{float:left}.actions[_ngcontent-%COMP%]{margin-top:16px}.actions[_ngcontent-%COMP%]:after{clear:both;content:"";display:table}input[_ngcontent-%COMP%]::-webkit-outer-spin-button, input[_ngcontent-%COMP%]::-webkit-inner-spin-button{-webkit-appearance:none;-moz-appearance:none;margin:0}.mat-mdc-form-field-hint[_ngcontent-%COMP%]{align-items:center;color:var(--theme-text-fade-30);display:flex;font-size:11px;margin-left:-10px;margin-right:-10px}.mdc-card[_ngcontent-%COMP%]{border:0}mat-error[_ngcontent-%COMP%]{font-size:10.5px;margin-left:-10px;margin-top:-3px}']
                        })
                    }
                    return n
                })(),
                Rs = (() => {
                    class n {
                        router;
                        ngZone;
                        addressId = void 0;
                        showNextButton = !0;
                        constructor(e, o) {
                            this.router = e, this.ngZone = o
                        }
                        getMessage(e) {
                            this.addressId = e
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(R.Ix), t.rXU(t.SKi))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-address-select"]
                            ],
                            decls: 2,
                            vars: 1,
                            consts: [
                                ["id", "card"],
                                [3, "showNextButton"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "div", 0), t.nrm(1, "app-address", 1), t.k0s()), 2 & o && (t.R7$(), t.Y8G("showNextButton", i.showNextButton))
                            },
                            dependencies: [Wt],
                            styles: ["\n\n\n\n.btn-pay[_ngcontent-%COMP%]{display:block;width:100%}.mat-mdc-raised-button[_ngcontent-%COMP%]{margin-right:3px;margin-top:3px}#card[_ngcontent-%COMP%]{display:block;margin-left:20%;margin-right:20%}"]
                        })
                    }
                    return n
                })(),
                ws = (() => {
                    class n {
                        static \u0275fac = function(o) {
                            return new(o || n)
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-saved-address"]
                            ],
                            decls: 2,
                            vars: 1,
                            consts: [
                                [1, "main-wrapper"],
                                [3, "allowEdit"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "div", 0), t.nrm(1, "app-address", 1), t.k0s()), 2 & o && (t.R7$(), t.Y8G("allowEdit", !0))
                            },
                            dependencies: [Wt],
                            styles: ["\n\n\n\n.main-wrapper[_ngcontent-%COMP%]{display:block;margin-left:20%;margin-right:20%}"]
                        })
                    }
                    return n
                })();
            S.Yv.add(C._sz);
            let Jt = (() => {
                    class n {
                        dialogData;
                        title;
                        url;
                        address;
                        data;
                        constructor(e) {
                            this.dialogData = e
                        }
                        ngOnInit() {
                            this.title = this.dialogData.title, this.url = this.dialogData.url, this.address = this.dialogData.address, this.data = this.dialogData.data
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(w.Vh))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-qr-code"]
                            ],
                            decls: 17,
                            vars: 11,
                            consts: [
                                [1, "qr-code"],
                                [1, "content-column"],
                                [1, "container"],
                                [3, "href"],
                                [3, "value", "size", "errorCorrectionLevel"],
                                ["mat-raised-button", "", "mat-dialog-close", "", 1, "close-dialog"],
                                [1, "fas", "fa-arrow-circle-left", "fa-lg"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "div", 0)(1, "header")(2, "h3"), t.EFF(3), t.nI1(4, "translate"), t.k0s()(), t.j41(5, "div", 1), t.nrm(6, "mat-divider"), t.j41(7, "div", 2)(8, "a", 3), t.nrm(9, "qr-code", 4), t.k0s()(), t.j41(10, "small"), t.EFF(11), t.k0s()(), t.j41(12, "footer")(13, "button", 5), t.nrm(14, "i", 6), t.EFF(15), t.nI1(16, "translate"), t.k0s()()()), 2 & o && (t.R7$(3), t.JRh(t.bMT(4, 7, i.title)), t.R7$(5), t.Y8G("href", i.url, t.B4B), t.R7$(), t.Y8G("value", i.data)("size", 300)("errorCorrectionLevel", "M"), t.R7$(2), t.JRh(i.address), t.R7$(4), t.SpI(" ", t.bMT(16, 9, "BTN_CLOSE"), " "))
                            },
                            dependencies: [U.q, pt.XK, pt.Um, b.Hl, b.$z, w.tx, d.h, d.D9],
                            styles: ["\n\n\n\n.qr-code[_ngcontent-%COMP%]{margin:16px}.content-column[_ngcontent-%COMP%]{display:flex;flex-direction:column;gap:10px}"]
                        })
                    }
                    return n
                })(),
                Qt = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/rest/wallet/balance";
                        constructor(e) {
                            this.http = e
                        }
                        get() {
                            return this.http.get(this.host).pipe((0, v.T)(e => e.data), (0, g.W)(e => {
                                throw e
                            }))
                        }
                        put(e) {
                            return this.http.put(this.host, e).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })(),
                Et = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/api/Deliverys";
                        constructor(e) {
                            this.http = e
                        }
                        get() {
                            return this.http.get(this.host).pipe((0, v.T)(e => e.data), (0, g.W)(e => {
                                throw e
                            }))
                        }
                        getById(e) {
                            return this.http.get(`${this.host}/${e}`).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })(),
                qt = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/api/Cards";
                        constructor(e) {
                            this.http = e
                        }
                        get() {
                            return this.http.get(this.host).pipe((0, v.T)(e => e.data), (0, g.W)(e => {
                                throw e
                            }))
                        }
                        getById(e) {
                            return this.http.get(`${this.host}/${e}`).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        save(e) {
                            return this.http.post(this.host + "/", e).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        del(e) {
                            return this.http.delete(`${this.host}/${e}`).pipe((0, v.T)(o => o.data), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })();

            function Es(n, a) {
                1 & n && t.nrm(0, "mat-header-cell")
            }

            function Ms(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-cell")(1, "mat-radio-button", 23), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG(2);
                        return s.Njj(r.emitSelectionToParent(i.id))
                    }), t.k0s()()
                }
            }

            function Is(n, a) {
                1 & n && t.nrm(0, "mat-header-cell")
            }

            function Os(n, a) {
                if (1 & n && (t.j41(0, "mat-cell"), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.JRh(e.cardNum)
                }
            }

            function Ps(n, a) {
                1 & n && t.nrm(0, "mat-header-cell")
            }

            function js(n, a) {
                if (1 & n && (t.j41(0, "mat-cell"), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.JRh(e.fullName)
                }
            }

            function As(n, a) {
                1 & n && t.nrm(0, "mat-header-cell")
            }

            function Ds(n, a) {
                if (1 & n && (t.j41(0, "mat-cell"), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.Lme("", e.expMonth, "/", e.expYear)
                }
            }

            function $s(n, a) {
                1 & n && t.nrm(0, "mat-header-cell")
            }

            function Ls(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-cell")(1, "button", 24), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG(2);
                        return s.Njj(r.delete(i.id))
                    }), t.nrm(2, "i", 25), t.k0s()()
                }
            }

            function Bs(n, a) {
                1 & n && t.nrm(0, "mat-header-row")
            }

            function Ns(n, a) {
                1 & n && t.nrm(0, "mat-row")
            }

            function Us(n, a) {
                if (1 & n && (t.j41(0, "div")(1, "mat-table", 13), t.qex(2, 14), t.DNE(3, Es, 1, 0, "mat-header-cell", 15)(4, Ms, 2, 0, "mat-cell", 16), t.bVm(), t.qex(5, 17), t.DNE(6, Is, 1, 0, "mat-header-cell", 15)(7, Os, 2, 1, "mat-cell", 16), t.bVm(), t.qex(8, 18), t.DNE(9, Ps, 1, 0, "mat-header-cell", 15)(10, js, 2, 1, "mat-cell", 16), t.bVm(), t.qex(11, 19), t.DNE(12, As, 1, 0, "mat-header-cell", 15)(13, Ds, 2, 2, "mat-cell", 16), t.bVm(), t.qex(14, 20), t.DNE(15, $s, 1, 0, "mat-header-cell", 15)(16, Ls, 3, 0, "mat-cell", 16), t.bVm(), t.DNE(17, Bs, 1, 0, "mat-header-row", 21)(18, Ns, 1, 0, "mat-row", 22), t.k0s()()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(), t.Y8G("dataSource", e.dataSource), t.R7$(16), t.Y8G("matHeaderRowDef", e.displayedColumns), t.R7$(), t.Y8G("matRowDefColumns", e.displayedColumns)
                }
            }

            function Vs(n, a) {
                1 & n && (t.j41(0, "mat-error", 5), t.EFF(1, " MANDATORY_NAME "), t.k0s())
            }

            function Gs(n, a) {
                1 & n && (t.j41(0, "mat-error", 5), t.EFF(1, " MANDATORY_CARD_NUMBER "), t.k0s())
            }

            function Ys(n, a) {
                1 & n && (t.j41(0, "mat-error", 5), t.EFF(1, " VALID_CARD_NUMBER "), t.k0s())
            }

            function Hs(n, a) {
                if (1 & n && (t.j41(0, "option", 11), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.Y8G("value", t.mNQ(e)), t.R7$(), t.JRh(e)
                }
            }

            function Xs(n, a) {
                1 & n && (t.j41(0, "mat-error", 5), t.EFF(1, " MANDATORY_EXPIRY_MONTH "), t.k0s())
            }

            function Ws(n, a) {
                if (1 & n && (t.j41(0, "option", 11), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.Y8G("value", t.mNQ(e)), t.R7$(), t.JRh(e)
                }
            }

            function zs(n, a) {
                1 & n && (t.j41(0, "mat-error", 5), t.EFF(1, " MANDATORY_EXPIRY_YEAR "), t.k0s())
            }
            S.Yv.add(C.isI, H.UB);
            let Ee = (() => {
                class n {
                    paymentService;
                    translate;
                    snackBarHelperService;
                    emitSelection = new t.bkB;
                    allowDelete = !1;
                    displayedColumns = ["Number", "Name", "Expiry"];
                    nameControl = new l.hs("", [l.k0.required]);
                    numberControl = new l.hs("", [l.k0.required, l.k0.min(1e15), l.k0.max(1e16)]);
                    monthControl = new l.hs("", [l.k0.required]);
                    yearControl = new l.hs("", [l.k0.required]);
                    confirmation;
                    error;
                    storedCards;
                    card = {};
                    dataSource;
                    monthRange;
                    yearRange;
                    cardsExist = !1;
                    paymentId = void 0;
                    constructor(e, o, i) {
                        this.paymentService = e, this.translate = o, this.snackBarHelperService = i
                    }
                    ngOnInit() {
                        this.monthRange = Array.from(Array(12).keys()).map(e => e + 1), this.yearRange = Array.from(Array(20).keys()).map(e => e + 2080), this.allowDelete ? this.displayedColumns.push("Remove") : this.displayedColumns.unshift("Selection"), this.load()
                    }
                    load() {
                        this.paymentService.get().subscribe({
                            next: e => {
                                this.cardsExist = e.length, this.storedCards = e, this.dataSource = new u.I6(this.storedCards)
                            },
                            error: e => {
                                console.log(e)
                            }
                        })
                    }
                    save() {
                        this.card.fullName = this.nameControl.value, this.card.cardNum = this.numberControl.value, this.card.expMonth = this.monthControl.value, this.card.expYear = this.yearControl.value, this.paymentService.save(this.card).subscribe({
                            next: e => {
                                this.error = null, this.translate.get("CREDIT_CARD_SAVED", {
                                    cardnumber: String(e.cardNum).substring(String(e.cardNum).length - 4)
                                }).subscribe({
                                    next: o => {
                                        this.snackBarHelperService.open(o, "confirmBar")
                                    },
                                    error: o => {
                                        this.snackBarHelperService.open(o, "confirmBar")
                                    }
                                }), this.load(), this.resetForm()
                            },
                            error: e => {
                                this.snackBarHelperService.open(e.error?.error, "errorBar"), this.resetForm()
                            }
                        })
                    }
                    delete(e) {
                        this.paymentService.del(e).subscribe({
                            next: () => {
                                this.load()
                            },
                            error: o => {
                                console.log(o)
                            }
                        })
                    }
                    emitSelectionToParent(e) {
                        this.emitSelection.emit(e)
                    }
                    resetForm() {
                        this.nameControl.markAsUntouched(), this.nameControl.markAsPristine(), this.nameControl.setValue(""), this.numberControl.markAsUntouched(), this.numberControl.markAsPristine(), this.numberControl.setValue(""), this.monthControl.markAsUntouched(), this.monthControl.markAsPristine(), this.monthControl.setValue(""), this.yearControl.markAsUntouched(), this.yearControl.markAsPristine(), this.yearControl.setValue("")
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(qt), t.rXU(d.c$), t.rXU(L.r))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-payment-method"]
                        ],
                        inputs: {
                            allowDelete: "allowDelete"
                        },
                        outputs: {
                            emitSelection: "emitSelection"
                        },
                        decls: 48,
                        vars: 24,
                        consts: [
                            ["cardIn", ""],
                            [1, "container"],
                            [1, "mat-elevation-z0"],
                            [2, "margin-bottom", "12px"],
                            ["appearance", "outline", "color", "accent"],
                            ["translate", ""],
                            ["type", "text", "matInput", "", 3, "formControl"],
                            ["type", "number", "matInput", "", 3, "formControl"],
                            ["align", "end"],
                            ["appearance", "outline", "color", "accent", 1, "half-field"],
                            ["matNativeControl", "", "required", "", 3, "formControl"],
                            [3, "value"],
                            ["type", "submit", "id", "submitButton", "mat-raised-button", "", "color", "primary", 2, "margin-top", "5px", "float", "right", 3, "click", "disabled"],
                            [3, "dataSource"],
                            ["matColumnDef", "Selection"],
                            [4, "matHeaderCellDef"],
                            [4, "matCellDef"],
                            ["matColumnDef", "Number"],
                            ["matColumnDef", "Name"],
                            ["matColumnDef", "Expiry"],
                            ["matColumnDef", "Remove"],
                            [4, "matHeaderRowDef"],
                            [4, "matRowDef", "matRowDefColumns"],
                            [3, "click"],
                            ["mat-icon-button", "", 3, "click"],
                            [1, "far", "fa-trash-alt"]
                        ],
                        template: function(o, i) {
                            if (1 & o) {
                                const r = t.RV6();
                                t.j41(0, "div", 1)(1, "h1"), t.EFF(2), t.nI1(3, "translate"), t.k0s(), t.nVh(4, Us, 19, 3, "div"), t.j41(5, "div")(6, "mat-expansion-panel", 2)(7, "mat-expansion-panel-header", 3)(8, "mat-panel-title"), t.EFF(9), t.nI1(10, "translate"), t.k0s(), t.j41(11, "mat-panel-description"), t.EFF(12), t.nI1(13, "translate"), t.k0s()(), t.j41(14, "div")(15, "mat-form-field", 4)(16, "mat-label", 5), t.EFF(17, "LABEL_NAME"), t.k0s(), t.nrm(18, "input", 6), t.nVh(19, Vs, 2, 0, "mat-error", 5), t.k0s(), t.j41(20, "mat-form-field", 4)(21, "mat-label", 5), t.EFF(22, "LABEL_CARD_NUMBER"), t.k0s(), t.nrm(23, "input", 7, 0), t.j41(25, "mat-hint", 8), t.EFF(26), t.k0s(), t.nVh(27, Gs, 2, 0, "mat-error", 5), t.nVh(28, Ys, 2, 0, "mat-error", 5), t.k0s(), t.j41(29, "mat-form-field", 9)(30, "mat-label", 5), t.EFF(31, "LABEL_EXPIRY_MONTH"), t.k0s(), t.j41(32, "select", 10), t.Z7z(33, Hs, 2, 3, "option", 11, t.fX1), t.k0s(), t.nVh(35, Xs, 2, 0, "mat-error", 5), t.k0s(), t.j41(36, "mat-form-field", 9)(37, "mat-label", 5), t.EFF(38, "LABEL_EXPIRY_YEAR"), t.k0s(), t.j41(39, "select", 10), t.Z7z(40, Ws, 2, 3, "option", 11, t.fX1), t.k0s(), t.nVh(42, zs, 2, 0, "mat-error", 5), t.k0s()(), t.j41(43, "button", 12), t.bIt("click", function() {
                                    return s.eBV(r), s.Njj(i.save())
                                }), t.j41(44, "mat-icon"), t.EFF(45, "send"), t.k0s(), t.EFF(46), t.nI1(47, "translate"), t.k0s()()()()
                            }
                            if (2 & o) {
                                const r = t.sdS(24);
                                t.R7$(2), t.JRh(t.bMT(3, 16, "MY_PAYMENT_OPTIONS")), t.R7$(2), t.vxM(i.cardsExist ? 4 : -1), t.R7$(5), t.SpI(" ", t.bMT(10, 18, "ADD_NEW_CARD"), " "), t.R7$(3), t.SpI(" ", t.bMT(13, 20, "ADD_NEW_CARD_DESCRIPTION"), " "), t.R7$(6), t.Y8G("formControl", i.nameControl), t.R7$(), t.vxM(i.nameControl.invalid && i.nameControl.errors.required ? 19 : -1), t.R7$(4), t.Y8G("formControl", i.numberControl), t.R7$(3), t.SpI("", (null == r.value ? null : r.value.length) || 0, "/16"), t.R7$(), t.vxM(i.numberControl.invalid && i.numberControl.errors.required ? 27 : -1), t.R7$(), t.vxM(i.numberControl.invalid && (i.numberControl.errors.min || i.numberControl.errors.max) ? 28 : -1), t.R7$(4), t.Y8G("formControl", i.monthControl), t.R7$(), t.Dyx(i.monthRange), t.R7$(2), t.vxM(i.monthControl.invalid && i.monthControl.errors.required ? 35 : -1), t.R7$(4), t.Y8G("formControl", i.yearControl), t.R7$(), t.Dyx(i.yearRange), t.R7$(2), t.vxM(i.yearControl.invalid && i.yearControl.errors.required ? 42 : -1), t.R7$(), t.Y8G("disabled", i.nameControl.invalid || i.numberControl.invalid || i.monthControl.invalid || i.yearControl.invalid), t.R7$(3), t.SpI(" ", t.bMT(47, 22, "BTN_SUBMIT"), " ")
                            }
                        },
                        dependencies: [u.Zl, u.cC, u.tL, u.KS, u.YV, u.$R, ot._g, V.iY, u.ji, u.YZ, u.iL, u.NB, G.GK, G.Z2, G.WN, G.Q6, I.R, _.rl, _.nJ, _.MV, _.TL, d.h, d.Mm, T.fS, T.fg, l.YN, l.xH, l.y7, l.me, l.Q0, l.wz, l.BC, l.YS, l.X1, l.l_, b.Hl, b.$z, f.m_, f.An, d.D9],
                        styles: ["\n\n\n\n.container[_ngcontent-%COMP%]{display:flex;flex-direction:column;gap:20px}mat-table[_ngcontent-%COMP%]{table-layout:fixed;width:100%}.mat-column-Selection[_ngcontent-%COMP%]{width:15%}.mat-column-Number[_ngcontent-%COMP%]{width:35%}.mat-column-Name[_ngcontent-%COMP%]{width:30%}.mat-column-Expiry[_ngcontent-%COMP%]{width:20%}.mat-column-Remove[_ngcontent-%COMP%]{width:15%}.half-field[_ngcontent-%COMP%]{margin-right:10%;width:45%}.half-field[_ngcontent-%COMP%]:last-of-type{margin-right:0}@media (max-width: 599.98px){.half-field[_ngcontent-%COMP%]{margin-right:0;width:100%}}.heading[_ngcontent-%COMP%]{background:#0003;font-size:x-large;justify-content:center;padding:12px 20px}mat-form-field[_ngcontent-%COMP%]{width:100%}.mat-expansion-panel-header-description[_ngcontent-%COMP%]{flex-basis:0}.detail-divider[_ngcontent-%COMP%]{margin-bottom:12px;margin-top:12px}mat-header-row[_ngcontent-%COMP%]{min-height:0!important}input[_ngcontent-%COMP%]::-webkit-outer-spin-button, input[_ngcontent-%COMP%]::-webkit-inner-spin-button{-webkit-appearance:none;-moz-appearance:none;margin:0}"]
                    })
                }
                return n
            })();
            const Js = (n, a) => ({
                    blueSky: n,
                    reddit: a
                }),
                Me = () => ({
                    juiceshop: "OWASP Juice Shop"
                }),
                Ie = n => ({
                    appname: n
                });

            function Qs(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "div", 4)(1, "div", 12)(2, "div", 13)(3, "span", 14), t.EFF(4, "LABEL_PAY_USING_WALLET"), t.k0s()(), t.j41(5, "div", 15)(6, "b")(7, "span", 14), t.EFF(8, "LABEL_WALLET_BALANCE"), t.k0s(), t.j41(9, "span", 16), t.EFF(10), t.k0s()()(), t.j41(11, "div", 17)(12, "button", 18), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.useWallet())
                    }), t.nrm(13, "i", 19), t.EFF(14), t.nI1(15, "translate"), t.k0s()()()()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.R7$(10), t.SpI(" ", e.walletBalanceStr), t.R7$(2), t.Y8G("disabled", e.walletBalance - e.totalPrice < 0), t.R7$(2), t.Lme(" ", t.bMT(15, 4, "LABEL_PAY"), " ", null == e.totalPrice ? null : e.totalPrice.toFixed(2), "\xa4 ")
                }
            }

            function qs(n, a) {
                if (1 & n && (t.j41(0, "div", 21), t.EFF(1), t.k0s()), 2 & n) {
                    const e = t.XpG(2);
                    t.R7$(), t.SpI("", e.couponConfirmation, " ")
                }
            }

            function Ks(n, a) {
                if (1 & n && (t.j41(0, "div", 22), t.EFF(1), t.k0s()), 2 & n) {
                    const e = t.XpG(2);
                    t.R7$(), t.SpI("", null == e.couponError ? null : e.couponError.error, " ")
                }
            }

            function Zs(n, a) {
                1 & n && (t.j41(0, "mat-error"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.SpI(" ", t.bMT(2, 1, "COUPON_CODE_HINT"), " "))
            }

            function tl(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-expansion-panel", 5)(1, "mat-expansion-panel-header", 20)(2, "mat-panel-title"), t.EFF(3), t.nI1(4, "translate"), t.k0s(), t.j41(5, "mat-panel-description"), t.EFF(6), t.nI1(7, "translate"), t.k0s()(), t.nVh(8, qs, 2, 1, "div", 21), t.nVh(9, Ks, 2, 1, "div", 22), t.j41(10, "mat-form-field", 23)(11, "mat-label", 10), t.EFF(12, "LABEL_COUPON"), t.k0s(), t.nrm(13, "mat-hint", 24), t.nI1(14, "translate"), t.nrm(15, "input", 25, 0), t.nI1(17, "translate"), t.j41(18, "mat-hint", 26), t.EFF(19), t.k0s(), t.nVh(20, Zs, 3, 3, "mat-error"), t.k0s(), t.j41(21, "button", 27), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.applyCoupon())
                    }), t.j41(22, "mat-icon"), t.EFF(23, " redeem "), t.k0s(), t.EFF(24), t.nI1(25, "translate"), t.k0s()()
                }
                if (2 & n) {
                    const e = t.sdS(16),
                        o = t.XpG();
                    t.Y8G("expanded", o.couponPanelExpanded), t.R7$(3), t.SpI(" ", t.bMT(4, 13, "ADD_A_COUPON"), " "), t.R7$(3), t.SpI(" ", t.bMT(7, 15, "VOUCHERS_AND_PROMOTIONAL_CODE"), " "), t.R7$(2), t.vxM(o.couponConfirmation && !o.couponControl.dirty ? 8 : -1), t.R7$(), t.vxM(o.couponError && !o.couponControl.dirty ? 9 : -1), t.R7$(4), t.Y8G("innerHtml", t.i5U(14, 17, "FOLLOW_FOR_MONTHLY_COUPONS", t.l_i(24, Js, o.blueSkyUrl, o.redditUrl)), t.npT), t.R7$(2), t.Y8G("placeholder", t.mNQ(t.bMT(17, 20, "ENTER_COUPON_CODE")))("formControl", o.couponControl), t.R7$(4), t.SpI("", (null == e.value ? null : e.value.length) || 0, "/10"), t.R7$(), t.vxM(o.couponControl.invalid && (o.couponControl.errors.minlength || o.couponControl.errors.maxlength) ? 20 : -1), t.R7$(), t.Y8G("disabled", o.couponControl.invalid), t.R7$(3), t.SpI(" ", t.bMT(25, 22, "BTN_REDEEM"), " ")
                }
            }

            function el(n, a) {
                1 & n && (t.j41(0, "span", 31), t.EFF(1, "THANKS_FOR_SUPPORT"), t.k0s()), 2 & n && t.Y8G("translateParams", t.lJ4(1, Me))
            }

            function nl(n, a) {
                if (1 & n && (t.j41(0, "span", 31), t.EFF(1, "THANKS_FOR_SUPPORT_CUSTOMIZED"), t.k0s()), 2 & n) {
                    const e = t.XpG(2);
                    t.Y8G("translateParams", t.eq3(1, Ie, e.applicationName))
                }
            }

            function ol(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 49), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG(2);
                        return s.Njj(i.showBitcoinQrCode())
                    }), t.nrm(1, "i", 50), t.EFF(2, " Bitcoin "), t.k0s()
                }
            }

            function il(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 49), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG(2);
                        return s.Njj(i.showDashQrCode())
                    }), t.j41(1, "i", 51), t.EFF(2, "\xd0"), t.k0s(), t.EFF(3, " Dash"), t.k0s()
                }
            }

            function al(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 49), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG(2);
                        return s.Njj(i.showEtherQrCode())
                    }), t.nrm(1, "i", 52), t.EFF(2, " Ether "), t.k0s()
                }
            }

            function rl(n, a) {
                1 & n && (t.j41(0, "span", 31), t.EFF(1, "OFFICIAL_MERCHANDISE_STORES"), t.k0s()), 2 & n && t.Y8G("translateParams", t.lJ4(1, Me))
            }

            function sl(n, a) {
                if (1 & n && (t.j41(0, "span", 31), t.EFF(1, "OFFICIAL_MERCHANDISE_STORES_CUSTOMIZED"), t.k0s()), 2 & n) {
                    const e = t.XpG(2);
                    t.Y8G("translateParams", t.eq3(1, Ie, e.applicationName))
                }
            }

            function ll(n, a) {
                if (1 & n && (t.j41(0, "mat-expansion-panel", 6)(1, "mat-expansion-panel-header", 20)(2, "mat-panel-title"), t.EFF(3), t.nI1(4, "translate"), t.k0s()(), t.j41(5, "div", 28)(6, "div", 29)(7, "div", 30)(8, "label", 10), t.EFF(9, "LABEL_DONATIONS"), t.k0s()(), t.j41(10, "small"), t.EFF(11, " ("), t.nVh(12, el, 2, 2, "span", 31)(13, nl, 2, 3, "span", 31), t.nrm(14, "i", 32), t.EFF(15, ") "), t.k0s(), t.j41(16, "div", 33)(17, "a", 34)(18, "button", 35), t.nrm(19, "i", 36), t.EFF(20), t.nI1(21, "translate"), t.k0s()(), t.nVh(22, ol, 3, 0, "button", 35), t.nVh(23, il, 4, 0, "button", 35), t.nVh(24, al, 3, 0, "button", 35), t.k0s()(), t.nrm(25, "span", 37), t.j41(26, "div", 38)(27, "div", 30)(28, "label", 10), t.EFF(29, "LABEL_MERCHANDISE"), t.k0s()(), t.j41(30, "small"), t.EFF(31, " ("), t.nVh(32, rl, 2, 2, "span", 31)(33, sl, 2, 3, "span", 31), t.nrm(34, "i", 39), t.EFF(35, ") "), t.k0s(), t.j41(36, "div", 33)(37, "a", 40)(38, "button", 35), t.nrm(39, "i", 41), t.EFF(40, " Spreadshirt (US)"), t.k0s()(), t.j41(41, "a", 42)(42, "button", 35), t.nrm(43, "i", 41), t.EFF(44, " Spreadshirt (DE)"), t.k0s()(), t.j41(45, "a", 43)(46, "button", 35), t.nrm(47, "i", 44), t.EFF(48, " StickerYou"), t.k0s()(), t.j41(49, "a", 45)(50, "button", 35), t.nrm(51, "i", 46), t.EFF(52, " Leanpub"), t.k0s()(), t.j41(53, "a", 47)(54, "button", 35), t.nrm(55, "i", 48), t.EFF(56, " OpenSea"), t.k0s()()()()()()), 2 & n) {
                    const e = t.XpG();
                    t.Y8G("expanded", e.paymentPanelExpanded), t.R7$(3), t.SpI(" ", t.bMT(4, 8, "OTHER_PAYMENT_OPTIONS"), " "), t.R7$(9), t.vxM("OWASP Juice Shop" === e.applicationName ? 12 : 13), t.R7$(8), t.SpI(" ", t.bMT(21, 10, "BTN_CREDIT_CARD")), t.R7$(2), t.vxM(-1), t.R7$(), t.vxM(-1), t.R7$(), t.vxM(-1), t.R7$(8), t.vxM("OWASP Juice Shop" === e.applicationName ? 32 : 33)
                }
            }

            function cl(n, a) {
                1 & n && (t.j41(0, "span", 10), t.EFF(1, "REVIEW_ALERT"), t.k0s())
            }

            function dl(n, a) {
                1 & n && (t.j41(0, "span", 10), t.EFF(1, "REVIEW_WALLET"), t.k0s())
            }
            S.Yv.add(C.Tu5, C.C97, C.qcK, N.t7R, C.Wcv, C.AdY, C.jyJ, C.eG7, C.euM, C.GRI, N.pw7, C.fyG);
            let ml = (() => {
                    class n {
                        location;
                        cookieService;
                        userService;
                        deliveryService;
                        walletService;
                        router;
                        dialog;
                        configurationService;
                        basketService;
                        translate;
                        activatedRoute;
                        ngZone;
                        snackBarHelperService;
                        couponConfirmation;
                        couponError;
                        card = {};
                        blueSkyUrl = null;
                        redditUrl = null;
                        applicationName = "OWASP Juice Shop";
                        campaignCoupon;
                        couponControl = new l.hs("", [l.k0.required, l.k0.minLength(10), l.k0.maxLength(10)]);
                        clientDate;
                        paymentId = void 0;
                        couponPanelExpanded = !1;
                        paymentPanelExpanded = !1;
                        mode;
                        walletBalance = 0;
                        walletBalanceStr;
                        totalPrice = 0;
                        paymentMode = "card";
                        campaigns = {
                            WMNSDY2019: {
                                validOn: 15519996e5,
                                discount: 75
                            },
                            WMNSDY2020: {
                                validOn: 1583622e6,
                                discount: 60
                            },
                            WMNSDY2021: {
                                validOn: 1615158e6,
                                discount: 60
                            },
                            WMNSDY2022: {
                                validOn: 1646694e6,
                                discount: 60
                            },
                            WMNSDY2023: {
                                validOn: 167823e7,
                                discount: 60
                            },
                            ORANGE2020: {
                                validOn: 15885468e5,
                                discount: 50
                            },
                            ORANGE2021: {
                                validOn: 16200828e5,
                                discount: 40
                            },
                            ORANGE2022: {
                                validOn: 16516188e5,
                                discount: 40
                            },
                            ORANGE2023: {
                                validOn: 16831548e5,
                                discount: 40
                            }
                        };
                        constructor(e, o, i, r, c, p, k, x, O, K, ht, gt, ft) {
                            this.location = e, this.cookieService = o, this.userService = i, this.deliveryService = r, this.walletService = c, this.router = p, this.dialog = k, this.configurationService = x, this.basketService = O, this.translate = K, this.activatedRoute = ht, this.ngZone = gt, this.snackBarHelperService = ft
                        }
                        ngOnInit() {
                            this.initTotal(), this.walletService.get().subscribe({
                                next: e => {
                                    this.walletBalance = e, this.walletBalanceStr = parseFloat(e).toFixed(2)
                                },
                                error: e => {
                                    console.log(e)
                                }
                            }), this.couponPanelExpanded = !!localStorage.getItem("couponPanelExpanded") && JSON.parse(localStorage.getItem("couponPanelExpanded")), this.paymentPanelExpanded = !!localStorage.getItem("paymentPanelExpanded") && JSON.parse(localStorage.getItem("paymentPanelExpanded")), this.configurationService.getApplicationConfiguration().subscribe({
                                next: e => {
                                    e?.application?.social && (e.application.social.blueSkyUrl && (this.blueSkyUrl = e.application.social.blueSkyUrl), e.application.social.redditUrl && (this.redditUrl = e.application.social.redditUrl), e.application.name && (this.applicationName = e.application.name))
                                },
                                error: e => {
                                    console.log(e)
                                }
                            })
                        }
                        initTotal() {
                            this.activatedRoute.paramMap.subscribe({
                                next: e => {
                                    if (this.mode = e.get("entity"), "wallet" === this.mode) this.totalPrice = parseFloat(sessionStorage.getItem("walletTotal"));
                                    else if ("deluxe" === this.mode) this.userService.deluxeStatus().subscribe({
                                        next: o => {
                                            this.totalPrice = o.membershipCost
                                        },
                                        error: o => {
                                            console.log(o)
                                        }
                                    });
                                    else {
                                        const o = parseFloat(sessionStorage.getItem("itemTotal")),
                                            i = sessionStorage.getItem("couponDiscount") ? parseFloat(sessionStorage.getItem("couponDiscount")) / 100 * o : 0;
                                        this.deliveryService.getById(sessionStorage.getItem("deliveryMethodId")).subscribe(r => {
                                            this.totalPrice = o + r.price - i
                                        })
                                    }
                                },
                                error: e => {
                                    console.log(e)
                                }
                            })
                        }
                        applyCoupon() {
                            this.campaignCoupon = this.couponControl.value, this.clientDate = new Date;
                            const e = 60 * (this.clientDate.getTimezoneOffset() + 60) * 1e3;
                            this.clientDate.setHours(0, 0, 0, 0), this.clientDate = this.clientDate.getTime() - e, sessionStorage.setItem("couponDetails", `${this.campaignCoupon}-${this.clientDate}`);
                            const o = this.campaigns[this.couponControl.value];
                            o ? this.clientDate === o.validOn ? this.showConfirmation(o.discount) : (this.couponConfirmation = void 0, this.translate.get("INVALID_COUPON").subscribe({
                                next: i => {
                                    this.couponError = {
                                        error: i
                                    }
                                },
                                error: i => {
                                    this.couponError = {
                                        error: i
                                    }
                                }
                            }), this.resetCouponForm()) : this.basketService.applyCoupon(Number(sessionStorage.getItem("bid")), encodeURIComponent(this.couponControl.value)).subscribe({
                                next: i => {
                                    this.showConfirmation(i)
                                },
                                error: i => {
                                    this.couponConfirmation = void 0, this.couponError = i, this.resetCouponForm()
                                }
                            })
                        }
                        showConfirmation(e) {
                            this.resetCouponForm(), this.couponError = void 0, sessionStorage.setItem("couponDiscount", e), this.translate.get("DISCOUNT_APPLIED", {
                                discount: e
                            }).subscribe({
                                next: o => {
                                    this.couponConfirmation = o
                                },
                                error: o => {
                                    this.couponConfirmation = o
                                }
                            }), this.initTotal()
                        }
                        getMessage(e) {
                            this.paymentId = e, this.paymentMode = "card"
                        }
                        routeToPreviousUrl() {
                            this.location.back()
                        }
                        choosePayment() {
                            var e = this;
                            if (sessionStorage.removeItem("itemTotal"), "wallet" === this.mode) this.walletService.put({
                                balance: this.totalPrice,
                                paymentId: this.paymentId
                            }).subscribe({
                                next: () => {
                                    sessionStorage.removeItem("walletTotal"), this.ngZone.run((0, F.A)(function*() {
                                        return yield e.router.navigate(["/wallet"])
                                    })), this.snackBarHelperService.open("CHARGED_WALLET", "confirmBar")
                                },
                                error: o => {
                                    console.log(o), this.snackBarHelperService.open(o.error?.message, "errorBar")
                                }
                            });
                            else if ("deluxe" === this.mode) this.userService.upgradeToDeluxe(this.paymentMode, this.paymentId).subscribe({
                                next: o => {
                                    localStorage.setItem("token", o.token), this.cookieService.put("token", o.token), this.ngZone.run((0, F.A)(function*() {
                                        return yield e.router.navigate(["/deluxe-membership"])
                                    }))
                                },
                                error: o => {
                                    console.log(o)
                                }
                            });
                            else {
                                if ("wallet" === this.paymentMode) {
                                    if (this.walletBalance < this.totalPrice) return void this.snackBarHelperService.open("INSUFFICIENT_WALLET_BALANCE", "errorBar");
                                    sessionStorage.setItem("paymentId", "wallet")
                                } else sessionStorage.setItem("paymentId", this.paymentId);
                                this.ngZone.run((0, F.A)(function*() {
                                    return yield e.router.navigate(["/order-summary"])
                                }))
                            }
                        }
                        noop() {}
                        showBitcoinQrCode() {
                            this.dialog.open(Jt, {
                                data: {
                                    data: "bitcoin:1AbKfgvw9psQ41NbLi8kufDQTezwG8DRZm",
                                    url: "./redirect?to=https://blockchain.info/address/1AbKfgvw9psQ41NbLi8kufDQTezwG8DRZm",
                                    address: "1AbKfgvw9psQ41NbLi8kufDQTezwG8DRZm",
                                    title: "TITLE_BITCOIN_ADDRESS"
                                }
                            })
                        }
                        showDashQrCode() {
                            this.dialog.open(Jt, {
                                data: {
                                    data: "dash:Xr556RzuwX6hg5EGpkybbv5RanJoZN17kW",
                                    url: "./redirect?to=https://explorer.dash.org/address/Xr556RzuwX6hg5EGpkybbv5RanJoZN17kW",
                                    address: "Xr556RzuwX6hg5EGpkybbv5RanJoZN17kW",
                                    title: "TITLE_DASH_ADDRESS"
                                }
                            })
                        }
                        showEtherQrCode() {
                            this.dialog.open(Jt, {
                                data: {
                                    data: "0x0f933ab9fCAAA782D0279C300D73750e1311EAE6",
                                    url: "./redirect?to=https://etherscan.io/address/0x0f933ab9fcaaa782d0279c300d73750e1311eae6",
                                    address: "0x0f933ab9fCAAA782D0279C300D73750e1311EAE6",
                                    title: "TITLE_ETHER_ADDRESS"
                                }
                            })
                        }
                        useWallet() {
                            this.paymentMode = "wallet", this.choosePayment()
                        }
                        resetCouponForm() {
                            this.couponControl.setValue(""), this.couponControl.markAsPristine(), this.couponControl.markAsUntouched()
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(zt.aZ), t.rXU($.O1), t.rXU(P), t.rXU(Et), t.rXU(Qt), t.rXU(R.Ix), t.rXU(w.bZ), t.rXU(M.s), t.rXU(Q), t.rXU(d.c$), t.rXU(R.nX), t.rXU(t.SKi), t.rXU(L.r))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-payment"]
                            ],
                            decls: 24,
                            vars: 10,
                            consts: [
                                ["coupon", ""],
                                ["appearance", "outlined", 1, "mat-own-card", "mat-elevation-z6"],
                                [1, "mdc-card"],
                                [3, "emitSelection", "allowDelete"],
                                [1, "mat-elevation-z0", "custom-card"],
                                ["id", "collapseCouponElement", 1, "mat-elevation-z0", 3, "expanded"],
                                [1, "mat-elevation-z0", 3, "expanded"],
                                [2, "margin-top", "20px", "margin-bottom", "20px"],
                                ["mat-stroked-button", "", 1, "btn", "btn-return", 3, "click"],
                                ["mat-raised-button", "", "mat-button", "", "color", "primary", "aria-label", "Proceed to review", 1, "btn", "nextButton", 3, "click", "disabled"],
                                ["translate", ""],
                                [2, "text-align", "center", "margin-top", "-27px"],
                                [1, "row"],
                                [1, "col", "col-42"],
                                ["translate", "", 1, "card-title"],
                                [1, "col", "col-38"],
                                [1, "confirmation", "card-title"],
                                [1, "col", "col-20"],
                                ["type", "submit", "color", "primary", "mat-raised-button", "", 1, "btn", 2, "float", "right", 3, "click", "disabled"],
                                [1, "fas", "fa-hand-holding-usd", "fa-lg"],
                                [1, "detail-divider"],
                                [1, "confirmation", 2, "margin-top", "5px"],
                                [1, "error", 2, "margin-top", "5px"],
                                ["appearance", "outline", "color", "accent"],
                                [3, "innerHtml"],
                                ["id", "coupon", "matInput", "", "type", "text", 3, "formControl", "placeholder"],
                                ["align", "end"],
                                ["type", "submit", "id", "applyCouponButton", "color", "accent", "mat-raised-button", "", 2, "margin-top", "5px", "float", "right", 3, "click", "disabled"],
                                [1, "row", "responsive-row"],
                                [1, "col", "col-34", "mat-elevation-z0"],
                                [1, "payment-label"],
                                ["translate", "", 3, "translateParams"],
                                [1, "fas", "fa-heart", "error", 2, "margin-left", "3px"],
                                [1, "button-container", 2, "margin-top", "6px"],
                                ["href", "https://pwning.owasp-juice.shop/companion-guide/latest/part3/donations.html"],
                                ["mat-stroked-button", ""],
                                [1, "fab", "fa-stripe", "fa-lg"],
                                [1, "fill-remaining-space"],
                                [1, "col", "col-65", "mat-elevation-z0"],
                                [1, "fas", "fa-thumbs-up", "confirmation", 2, "margin-left", "3px"],
                                ["href", "./redirect?to=http://shop.spreadshirt.com/juiceshop"],
                                [1, "fas", "fa-tshirt", "fa-lg"],
                                ["href", "./redirect?to=http://shop.spreadshirt.de/juiceshop"],
                                ["href", "./redirect?to=https://www.stickeryou.com/products/owasp-juice-shop/794"],
                                [1, "fas", "fa-sticky-note", "fa-lg"],
                                ["href", "./redirect?to=http://leanpub.com/juice-shop"],
                                [1, "fab", "fa-leanpub", "fa-lg"],
                                ["href", "https://opensea.io/collection/juice-shop"],
                                [1, "fas", "fa-palette", "fa-lg"],
                                ["mat-stroked-button", "", 3, "click"],
                                [1, "fab", "fa-btc", "fa-lg"],
                                [1, "fa-lg"],
                                [1, "fab", "fa-ethereum", "fa-lg"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "mat-card", 1)(1, "div", 2)(2, "app-payment-method", 3), t.bIt("emitSelection", function(c) {
                                    return i.getMessage(c)
                                }), t.k0s(), t.nrm(3, "mat-divider"), t.nVh(4, Qs, 16, 6, "div", 4), t.nrm(5, "mat-divider"), t.nVh(6, tl, 26, 27, "mat-expansion-panel", 5), t.nrm(7, "mat-divider"), t.nVh(8, ll, 57, 12, "mat-expansion-panel", 6), t.nrm(9, "mat-divider"), t.j41(10, "div", 7)(11, "button", 8), t.bIt("click", function() {
                                    return i.routeToPreviousUrl()
                                }), t.j41(12, "mat-icon"), t.EFF(13, " navigate_before "), t.k0s(), t.EFF(14), t.nI1(15, "translate"), t.k0s(), t.j41(16, "button", 9), t.bIt("click", function() {
                                    return i.choosePayment()
                                }), t.j41(17, "mat-icon"), t.EFF(18, " navigate_next "), t.k0s(), t.j41(19, "span", 10), t.EFF(20, "LABEL_CONTINUE"), t.k0s()(), t.j41(21, "p", 11), t.nVh(22, cl, 2, 0, "span", 10), t.nVh(23, dl, 2, 0, "span", 10), t.k0s()()()()), 2 & o && (t.R7$(2), t.Y8G("allowDelete", !1), t.R7$(2), t.vxM("wallet" !== i.mode ? 4 : -1), t.R7$(2), t.vxM("wallet" !== i.mode ? 6 : -1), t.R7$(2), t.vxM("wallet" !== i.mode ? 8 : -1), t.R7$(6), t.SpI(" ", t.bMT(15, 8, "LABEL_BACK"), " "), t.R7$(2), t.Y8G("disabled", void 0 === i.paymentId && "wallet" !== i.paymentMode), t.R7$(6), t.vxM("deluxe" !== i.mode && "wallet" !== i.mode ? 22 : -1), t.R7$(), t.vxM("wallet" === i.mode ? 23 : -1))
                            },
                            dependencies: [h.Hu, h.RN, Ee, U.q, d.h, d.Mm, b.Hl, b.$z, G.GK, G.Z2, G.WN, G.Q6, I.R, _.rl, _.nJ, _.MV, _.TL, T.fS, T.fg, l.YN, l.me, l.BC, l.X1, l.l_, f.m_, f.An, d.D9],
                            styles: ["\n\n\n\nmat-form-field[_ngcontent-%COMP%]{width:100%}.mat-own-card[_ngcontent-%COMP%]{display:block;margin-left:20%;margin-right:20%}.btn-return[_ngcontent-%COMP%]{display:block;margin-left:0;margin-right:auto}.nextButton[_ngcontent-%COMP%]{display:block;margin-left:auto;margin-right:0;margin-top:-37px}mat-card[_ngcontent-%COMP%]{height:auto}.mat-expansion-panel-header-description[_ngcontent-%COMP%]{flex-basis:0}.mat-mdc-raised-button[_ngcontent-%COMP%]{margin-right:3px}.custom-card[_ngcontent-%COMP%]{padding:18px 24px}.card-title[_ngcontent-%COMP%]{font-size:14px;line-height:36px;vertical-align:middle}.fill-remaining-space[_ngcontent-%COMP%]{flex:1 1 auto}.detail-divider[_ngcontent-%COMP%]{margin-bottom:12px;margin-top:12px}.mat-mdc-card[_ngcontent-%COMP%]   .mat-divider-horizontal[_ngcontent-%COMP%]{position:relative!important}.row[_ngcontent-%COMP%]{align-items:center;display:flex}.col[_ngcontent-%COMP%]{min-width:0}.col-42[_ngcontent-%COMP%]{flex:0 0 42%;max-width:42%}.col-38[_ngcontent-%COMP%]{flex:0 0 38%;max-width:38%}.col-20[_ngcontent-%COMP%]{flex:0 0 20%;max-width:20%}.col-34[_ngcontent-%COMP%]{flex:0 0 34%;max-width:34%}.col-65[_ngcontent-%COMP%]{flex:0 0 65%;max-width:65%}.responsive-row[_ngcontent-%COMP%]{flex-wrap:nowrap}@media (max-width: 1279.98px){.responsive-row[_ngcontent-%COMP%]{flex-direction:column}.responsive-row[_ngcontent-%COMP%]   .col[_ngcontent-%COMP%]{flex:0 0 100%;max-width:100%}.fill-remaining-space[_ngcontent-%COMP%]{display:none}}"]
                        })
                    }
                    return n
                })(),
                pl = (() => {
                    class n {
                        static \u0275fac = function(o) {
                            return new(o || n)
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["app-saved-payment-methods"]
                            ],
                            decls: 3,
                            vars: 1,
                            consts: [
                                ["appearance", "outlined", 1, "mat-own-card", "mat-elevation-z6"],
                                [1, "mdc-card"],
                                [3, "allowDelete"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "mat-card", 0)(1, "div", 1), t.nrm(2, "app-payment-method", 2), t.k0s()()), 2 & o && (t.R7$(2), t.Y8G("allowDelete", !0))
                            },
                            dependencies: [h.Hu, h.RN, Ee],
                            styles: ["\n\n\n\n.mat-own-card[_ngcontent-%COMP%]{display:block;margin-left:20%;margin-right:20%}"]
                        })
                    }
                    return n
                })(),
                Kt = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/rest/order-history";
                        constructor(e) {
                            this.http = e
                        }
                        get() {
                            return this.http.get(this.host).pipe((0, v.T)(e => e.data), (0, g.W)(e => {
                                throw e
                            }))
                        }
                        getAll() {
                            return this.http.get(this.host + "/orders").pipe((0, v.T)(e => e.data), (0, g.W)(e => {
                                throw e
                            }))
                        }
                        toggleDeliveryStatus(e, o) {
                            return this.http.put(`${this.host}/${e}/delivery-status`, o).pipe((0, v.T)(i => i.data), (0, g.W)(i => {
                                throw i
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })();
            const ul = ["paginatorOrderHistory"],
                hl = ["paginator"];

            function gl(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 30)
            }

            function fl(n, a) {
                if (1 & n && (t.j41(0, "mat-cell"), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.SpI(" ", e.orderId)
                }
            }

            function _l(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 31)
            }

            function Cl(n, a) {
                if (1 & n && (t.j41(0, "mat-cell", 32), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.SpI(" ", null == e.totalPrice ? null : e.totalPrice.toFixed(2), "\xa4")
                }
            }

            function bl(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 33)
            }

            function vl(n, a) {
                1 & n && (t.j41(0, "div", 34), t.EFF(1, "LABEL_DELIVERED"), t.k0s())
            }

            function kl(n, a) {
                1 & n && (t.j41(0, "div", 35), t.EFF(1, "LABEL_IN_TRANSIT"), t.k0s())
            }

            function Sl(n, a) {
                if (1 & n && (t.j41(0, "mat-cell", 32), t.nVh(1, vl, 2, 0, "div", 34)(2, kl, 2, 0, "div", 35), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.vxM(e.delivered ? 1 : 2)
                }
            }

            function Fl(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 32)
            }

            function xl(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 37), t.nI1(1, "translate"), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG().$implicit,
                            r = t.XpG();
                        return s.Njj(r.changeDeliveryStatus(i.delivered, i.id))
                    }), t.j41(2, "mat-icon"), t.EFF(3, " cached "), t.k0s()()
                }
                2 & n && t.Y8G("matTooltip", t.mNQ(t.bMT(1, 2, "LABEL_MARK_AS_TRANSIT")))
            }

            function Tl(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 37), t.nI1(1, "translate"), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG().$implicit,
                            r = t.XpG();
                        return s.Njj(r.changeDeliveryStatus(i.delivered, i.id))
                    }), t.j41(2, "mat-icon"), t.EFF(3, " check_circle "), t.k0s()()
                }
                2 & n && t.Y8G("matTooltip", t.mNQ(t.bMT(1, 2, "LABEL_MARK_AS_DELIVERED")))
            }

            function yl(n, a) {
                if (1 & n && (t.j41(0, "mat-cell", 32), t.nVh(1, xl, 4, 4, "button", 36)(2, Tl, 4, 4, "button", 36), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.vxM(e.delivered ? 1 : 2)
                }
            }

            function Rl(n, a) {
                1 & n && t.nrm(0, "mat-header-row")
            }

            function wl(n, a) {
                1 & n && t.nrm(0, "mat-row")
            }

            function El(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 38)
            }

            function Ml(n, a) {
                if (1 & n && (t.j41(0, "mat-cell"), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.JRh(e.name)
                }
            }

            function Il(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 39)
            }

            function Ol(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-cell")(1, "mat-form-field", 40), t.nrm(2, "input", 41, 2), t.j41(4, "button", 42), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.sdS(3),
                            c = t.XpG();
                        return s.Njj(c.modifyPrice(i.id, r.value))
                    }), t.nrm(5, "i", 43), t.k0s()()()
                }
                if (2 & n) {
                    const e = a.$implicit;
                    t.R7$(2), t.Y8G("value", t.mNQ(e.price))
                }
            }

            function Pl(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 44)
            }

            function jl(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-cell")(1, "mat-form-field", 40), t.nrm(2, "input", 41, 3), t.j41(4, "button", 42), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.sdS(3),
                            c = t.XpG();
                        return s.Njj(c.modifyQuantity(c.quantityMap[i.id].id, r.value))
                    }), t.nrm(5, "i", 43), t.k0s()()()
                }
                if (2 & n) {
                    const e = a.$implicit,
                        o = t.XpG();
                    t.R7$(2), t.Y8G("value", t.mNQ(o.quantityMap[e.id].quantity))
                }
            }

            function Al(n, a) {
                1 & n && t.nrm(0, "mat-header-row")
            }

            function Dl(n, a) {
                1 & n && t.nrm(0, "mat-row")
            }
            S.Yv.add(C.e68);
            let $l = (() => {
                class n {
                    productService;
                    quantityService;
                    orderHistoryService;
                    snackBarHelperService;
                    orderHistoryColumns = ["OrderId", "Price", "Status", "StatusButton"];
                    paginatorOrderHistory;
                    orderData;
                    orderSource;
                    displayedColumns = ["Product", "Price", "Quantity"];
                    tableData;
                    dataSource;
                    paginator;
                    productSubscription;
                    quantitySubscription;
                    quantityMap;
                    constructor(e, o, i, r) {
                        this.productService = e, this.quantityService = o, this.orderHistoryService = i, this.snackBarHelperService = r
                    }
                    ngAfterViewInit() {
                        this.loadQuantity(), this.loadProducts(), this.loadOrders()
                    }
                    loadQuantity() {
                        this.quantitySubscription = this.quantityService.getAll().subscribe({
                            next: e => {
                                this.quantityMap = {}, e.forEach(o => {
                                    this.quantityMap[o.ProductId] = {
                                        id: o.id,
                                        quantity: o.quantity
                                    }
                                })
                            },
                            error: e => {
                                console.log(e)
                            }
                        })
                    }
                    loadProducts() {
                        this.productSubscription = this.productService.search("").subscribe({
                            next: e => {
                                this.tableData = e, this.dataSource = new u.I6(this.tableData), this.dataSource.paginator = this.paginator
                            },
                            error: e => {
                                console.log(e)
                            }
                        })
                    }
                    loadOrders() {
                        this.orderHistoryService.getAll().subscribe({
                            next: e => {
                                this.orderData = [];
                                for (const o of e) this.orderData.push({
                                    id: o._id,
                                    orderId: o.orderId,
                                    totalPrice: o.totalPrice,
                                    delivered: o.delivered
                                });
                                this.orderSource = new u.I6(this.orderData), this.orderSource.paginator = this.paginatorOrderHistory
                            },
                            error: e => {
                                console.log(e)
                            }
                        })
                    }
                    ngOnDestroy() {
                        this.productSubscription && this.productSubscription.unsubscribe(), this.quantitySubscription && this.quantitySubscription.unsubscribe()
                    }
                    modifyQuantity(e, o) {
                        this.quantityService.put(e, {
                            quantity: o < 0 ? 0 : o
                        }).subscribe({
                            next: i => {
                                const r = this.tableData.find(c => c.id === i.ProductId);
                                this.snackBarHelperService.open(`Quantity for ${r.name} has been updated.`, "confirmBar"), this.loadQuantity()
                            },
                            error: i => {
                                this.snackBarHelperService.open(i.error, "errorBar"), console.log(i)
                            }
                        })
                    }
                    modifyPrice(e, o) {
                        this.productService.put(e, {
                            price: o < 0 ? 0 : o
                        }).subscribe({
                            next: i => {
                                this.snackBarHelperService.open(`Price for ${i.name} has been updated.`, "confirmBar"), this.loadProducts()
                            },
                            error: i => {
                                this.snackBarHelperService.open(i.error, "errorBar"), console.log(i)
                            }
                        })
                    }
                    changeDeliveryStatus(e, o) {
                        this.orderHistoryService.toggleDeliveryStatus(o, {
                            deliveryStatus: e
                        }).subscribe({
                            next: () => {
                                this.loadOrders()
                            },
                            error: i => {
                                this.snackBarHelperService.open(i, "errorBar"), console.log(i)
                            }
                        })
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(Rt), t.rXU(Ht), t.rXU(Kt), t.rXU(L.r))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-accounting"]
                        ],
                        viewQuery: function(o, i) {
                            if (1 & o && (t.GBs(ul, 7), t.GBs(hl, 7)), 2 & o) {
                                let r;
                                t.mGM(r = t.lsd()) && (i.paginatorOrderHistory = r.first), t.mGM(r = t.lsd()) && (i.paginator = r.first)
                            }
                        },
                        decls: 44,
                        vars: 8,
                        consts: [
                            ["paginatorOrderHistory", ""],
                            ["paginator", ""],
                            ["price", ""],
                            ["quanitity", ""],
                            ["appearance", "outlined", 1, "mat-elevation-z6", "mat-own-card"],
                            [1, "mdc-card"],
                            ["translate", ""],
                            [1, "container"],
                            [1, "orders-table"],
                            [1, "heading"],
                            [3, "dataSource"],
                            ["matColumnDef", "OrderId"],
                            ["translate", "LABEL_ORDER_ID", 4, "matHeaderCellDef"],
                            [4, "matCellDef"],
                            ["matColumnDef", "Price"],
                            ["translate", "LABEL_PRICE", "style", "max-width: 96px;", 4, "matHeaderCellDef"],
                            ["style", "max-width: 96px;", 4, "matCellDef"],
                            ["matColumnDef", "Status"],
                            ["translate", "LABEL_STATUS", "style", "max-width: 96px;", 4, "matHeaderCellDef"],
                            ["matColumnDef", "StatusButton"],
                            ["style", "max-width: 96px;", 4, "matHeaderCellDef"],
                            [4, "matHeaderRowDef"],
                            [4, "matRowDef", "matRowDefColumns"],
                            ["color", "accent", 1, "mat-elevation-z0", 3, "pageSize"],
                            [1, "inventory-table"],
                            ["matColumnDef", "Product"],
                            ["translate", "LABEL_PRODUCT", 4, "matHeaderCellDef"],
                            ["translate", "LABEL_PRICE", 4, "matHeaderCellDef"],
                            ["matColumnDef", "Quantity"],
                            ["translate", "LABEL_QUANTITY", 4, "matHeaderCellDef"],
                            ["translate", "LABEL_ORDER_ID"],
                            ["translate", "LABEL_PRICE", 2, "max-width", "96px"],
                            [2, "max-width", "96px"],
                            ["translate", "LABEL_STATUS", 2, "max-width", "96px"],
                            ["translate", "", 1, "confirmation"],
                            ["translate", "", 1, "error"],
                            ["mat-icon-button", "", "aria-label", "Print order confirmation", "matTooltipPosition", "below", 3, "matTooltip"],
                            ["mat-icon-button", "", "aria-label", "Print order confirmation", "matTooltipPosition", "below", 3, "click", "matTooltip"],
                            ["translate", "LABEL_PRODUCT"],
                            ["translate", "LABEL_PRICE"],
                            [1, "input-field"],
                            ["matInput", "", "type", "number", 3, "value"],
                            ["mat-icon-button", "", "matSuffix", "", 3, "click"],
                            [1, "fas", "fa-check"],
                            ["translate", "LABEL_QUANTITY"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "mat-card", 4)(1, "div", 5)(2, "h1", 6), t.EFF(3, "ACCOUNTING"), t.k0s(), t.j41(4, "div", 7)(5, "div", 8)(6, "div", 9)(7, "span", 6), t.EFF(8, "TITLE_TRACK_ORDERS"), t.k0s()(), t.j41(9, "mat-table", 10), t.qex(10, 11), t.DNE(11, gl, 1, 0, "mat-header-cell", 12)(12, fl, 2, 1, "mat-cell", 13), t.bVm(), t.qex(13, 14), t.DNE(14, _l, 1, 0, "mat-header-cell", 15)(15, Cl, 2, 1, "mat-cell", 16), t.bVm(), t.qex(16, 17), t.DNE(17, bl, 1, 0, "mat-header-cell", 18)(18, Sl, 3, 1, "mat-cell", 16), t.bVm(), t.qex(19, 19), t.DNE(20, Fl, 1, 0, "mat-header-cell", 20)(21, yl, 3, 1, "mat-cell", 16), t.bVm(), t.DNE(22, Rl, 1, 0, "mat-header-row", 21)(23, wl, 1, 0, "mat-row", 22), t.k0s(), t.nrm(24, "mat-paginator", 23, 0), t.k0s(), t.j41(26, "div", 24)(27, "div", 9)(28, "span", 6), t.EFF(29, "TITLE_ALL_PRODUCTS"), t.k0s()(), t.j41(30, "mat-table", 10), t.qex(31, 25), t.DNE(32, El, 1, 0, "mat-header-cell", 26)(33, Ml, 2, 1, "mat-cell", 13), t.bVm(), t.qex(34, 14), t.DNE(35, Il, 1, 0, "mat-header-cell", 27)(36, Ol, 6, 2, "mat-cell", 13), t.bVm(), t.qex(37, 28), t.DNE(38, Pl, 1, 0, "mat-header-cell", 29)(39, jl, 6, 2, "mat-cell", 13), t.bVm(), t.DNE(40, Al, 1, 0, "mat-header-row", 21)(41, Dl, 1, 0, "mat-row", 22), t.k0s(), t.nrm(42, "mat-paginator", 23, 1), t.k0s()()()()), 2 & o && (t.R7$(9), t.Y8G("dataSource", i.orderSource), t.R7$(13), t.Y8G("matHeaderRowDef", i.orderHistoryColumns), t.R7$(), t.Y8G("matRowDefColumns", i.orderHistoryColumns), t.R7$(), t.Y8G("pageSize", 10), t.R7$(6), t.Y8G("dataSource", i.dataSource), t.R7$(10), t.Y8G("matHeaderRowDef", i.displayedColumns), t.R7$(), t.Y8G("matRowDefColumns", i.displayedColumns), t.R7$(), t.Y8G("pageSize", 10))
                        },
                        dependencies: [h.Hu, h.RN, d.h, d.Mm, u.Zl, u.cC, u.tL, u.KS, u.YV, u.$R, V.iY, B.oV, f.m_, f.An, u.ji, u.YZ, u.iL, u.NB, mt.iy, I.R, _.rl, _.yw, T.fS, T.fg, d.D9],
                        styles: ["\n\n\n\n.heading[_ngcontent-%COMP%]{background:#00000021;font-size:large;justify-content:center;margin-bottom:10px;padding:12px}.container[_ngcontent-%COMP%]{display:grid;gap:48px;grid-template-columns:1fr 1fr}@media (max-width: 1400px){.container[_ngcontent-%COMP%]{gap:24px;grid-template-columns:1fr}}mat-table[_ngcontent-%COMP%]{height:960px}mat-header-row[_ngcontent-%COMP%]{min-height:0!important}.orders-table[_ngcontent-%COMP%], .inventory-table[_ngcontent-%COMP%]{margin-bottom:25px}mat-header-cell[_ngcontent-%COMP%]{padding-bottom:4px}mat-cell[_ngcontent-%COMP%], mat-footer-cell[_ngcontent-%COMP%], mat-header-cell[_ngcontent-%COMP%]{align-items:center;display:flex}mat-row[_ngcontent-%COMP%]{align-items:center;display:flex;padding:4px 0}.mat-own-card[_ngcontent-%COMP%]{display:block;margin-bottom:20px;margin-left:10%;margin-right:10%}.product[_ngcontent-%COMP%]{cursor:pointer}.input-field[_ngcontent-%COMP%]{font-size:14px;width:160px}"]
                    })
                }
                return n
            })();
            const Ll = n => ({
                    id: n
                }),
                Bl = n => ({
                    bonus: n
                }),
                Nl = n => ({
                    numberdays: n
                });

            function Ul(n, a) {
                if (1 & n && (t.j41(0, "span")(1, "div", 27), t.EFF(2, "ESTIMATED_TIME_OF_DELIVERY"), t.k0s()()), 2 & n) {
                    const e = t.XpG(2);
                    t.R7$(), t.Y8G("translateParams", t.eq3(1, Nl, e.orderDetails.eta))
                }
            }

            function Vl(n, a) {
                if (1 & n && (t.j41(0, "div"), t.nVh(1, Ul, 3, 3, "span"), t.j41(2, "div")(3, "b", 4), t.EFF(4, "LABEL_DELIVERY_ADDRESS"), t.k0s()(), t.j41(5, "div"), t.EFF(6), t.k0s(), t.j41(7, "div"), t.EFF(8), t.k0s(), t.j41(9, "div"), t.EFF(10), t.k0s(), t.j41(11, "div")(12, "span", 4), t.EFF(13, "PHONE_NUMBER"), t.k0s(), t.EFF(14), t.k0s()()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(), t.vxM("?" !== e.orderDetails.eta && void 0 !== e.orderDetails.eta ? 1 : -1), t.R7$(5), t.JRh(null == e.address ? null : e.address.fullName), t.R7$(2), t.LHq("", null == e.address ? null : e.address.streetAddress, ", ", null == e.address ? null : e.address.city, ", ", null == e.address ? null : e.address.state, ", ", null == e.address ? null : e.address.zipCode), t.R7$(2), t.JRh(null == e.address ? null : e.address.country), t.R7$(4), t.SpI(" ", null == e.address ? null : e.address.mobileNum)
                }
            }

            function Gl(n, a) {
                1 & n && (t.j41(0, "mat-header-cell", 4), t.EFF(1, "LABEL_PRODUCT"), t.k0s())
            }

            function Yl(n, a) {
                if (1 & n && (t.j41(0, "mat-cell"), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.SpI("", e.name, " ")
                }
            }

            function Hl(n, a) {
                1 & n && t.nrm(0, "mat-footer-cell")
            }

            function Xl(n, a) {
                1 & n && (t.j41(0, "mat-header-cell", 4), t.EFF(1, "LABEL_PRICE"), t.k0s())
            }

            function Wl(n, a) {
                if (1 & n && (t.j41(0, "mat-cell"), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.SpI("", e.price, "\xa4")
                }
            }

            function zl(n, a) {
                1 & n && t.nrm(0, "mat-footer-cell")
            }

            function Jl(n, a) {
                1 & n && (t.j41(0, "mat-header-cell", 4), t.EFF(1, "LABEL_QUANTITY"), t.k0s())
            }

            function Ql(n, a) {
                if (1 & n && (t.j41(0, "mat-cell")(1, "span"), t.EFF(2), t.k0s()()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(2), t.JRh(e.quantity)
                }
            }

            function ql(n, a) {
                1 & n && (t.j41(0, "mat-footer-cell")(1, "table", 28)(2, "tr", 29)(3, "td", 4), t.EFF(4, "ITEMS"), t.k0s()(), t.j41(5, "tr", 29)(6, "td", 4), t.EFF(7, "DELIVERY"), t.k0s()(), t.j41(8, "tr", 29)(9, "td", 4), t.EFF(10, "PROMOTION"), t.k0s()(), t.j41(11, "tr", 30)(12, "td", 4), t.EFF(13, "LABEL_TOTAL_PRICE"), t.k0s()()()())
            }

            function Kl(n, a) {
                1 & n && (t.j41(0, "mat-header-cell", 4), t.EFF(1, "LABEL_TOTAL_PRICE"), t.k0s())
            }

            function Zl(n, a) {
                if (1 & n && (t.j41(0, "mat-cell", 31), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.SpI("", e.total.toFixed(2), "\xa4")
                }
            }

            function tc(n, a) {
                if (1 & n && (t.j41(0, "mat-footer-cell")(1, "table", 31)(2, "tr", 29)(3, "td"), t.EFF(4), t.k0s()(), t.j41(5, "tr", 29)(6, "td"), t.EFF(7), t.k0s()(), t.j41(8, "tr", 29)(9, "td"), t.EFF(10), t.k0s()(), t.j41(11, "tr", 30)(12, "td"), t.EFF(13), t.k0s()()()()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(4), t.SpI("", null == e.orderDetails.itemTotal ? null : e.orderDetails.itemTotal.toFixed(2), "\xa4"), t.R7$(3), t.SpI("", e.deliveryPrice.toFixed(2), "\xa4"), t.R7$(3), t.SpI("", e.promotionalDiscount.toFixed(2), "\xa4"), t.R7$(3), t.SpI("", null == e.orderDetails.totalPrice ? null : e.orderDetails.totalPrice.toFixed(2), "\xa4")
                }
            }

            function ec(n, a) {
                1 & n && t.nrm(0, "mat-header-row")
            }

            function nc(n, a) {
                1 & n && t.nrm(0, "mat-row")
            }

            function oc(n, a) {
                1 & n && t.nrm(0, "mat-footer-row", 32)
            }
            S.Yv.add(N.HQ1);
            let ic = (() => {
                class n {
                    configurationService;
                    addressService;
                    trackOrderService;
                    activatedRoute;
                    basketService;
                    tableColumns = ["product", "price", "quantity", "total price"];
                    dataSource;
                    orderId;
                    orderDetails = {
                        totalPrice: 0
                    };
                    deliveryPrice = 0;
                    promotionalDiscount = 0;
                    address;
                    tweetText = "I just purchased";
                    constructor(e, o, i, r, c) {
                        this.configurationService = e, this.addressService = o, this.trackOrderService = i, this.activatedRoute = r, this.basketService = c
                    }
                    ngOnInit() {
                        this.activatedRoute.paramMap.subscribe({
                            next: e => {
                                this.orderId = e.get("id"), this.trackOrderService.find(this.orderId).subscribe({
                                    next: o => {
                                        this.promotionalDiscount = o.data[0].promotionalAmount ? parseFloat(o.data[0].promotionalAmount) : 0, this.deliveryPrice = o.data[0].deliveryPrice ? parseFloat(o.data[0].deliveryPrice) : 0, this.orderDetails.addressId = o.data[0].addressId, this.orderDetails.paymentId = o.data[0].paymentId, this.orderDetails.totalPrice = o.data[0].totalPrice, this.orderDetails.itemTotal = o.data[0].totalPrice + this.promotionalDiscount - this.deliveryPrice, this.orderDetails.eta = o.data[0].eta || "?", this.orderDetails.products = o.data[0].products, this.orderDetails.bonus = o.data[0].bonus, this.dataSource = new u.I6(this.orderDetails.products);
                                        for (const i of this.orderDetails.products) this.tweetText += `%0a- ${i.name}`;
                                        this.tweetText = this.truncateTweet(this.tweetText), this.configurationService.getApplicationConfiguration().subscribe({
                                            next: i => {
                                                i?.application?.social && (this.tweetText += "%0afrom ", this.tweetText += i.application.social.twitterUrl ? i.application.social.twitterUrl.replace("https://twitter.com/", "@") : i.application.name)
                                            },
                                            error: i => {
                                                console.log(i)
                                            }
                                        }), this.addressService.getById(this.orderDetails.addressId).subscribe({
                                            next: i => {
                                                this.address = i
                                            },
                                            error: i => {
                                                console.log(i)
                                            }
                                        })
                                    },
                                    error: o => {
                                        console.log(o)
                                    }
                                })
                            },
                            error: e => {
                                console.log(e)
                            }
                        })
                    }
                    openConfirmationPDF() {
                        window.open(`${this.basketService.hostServer}/ftp/order_${this.orderId}.pdf`, "_blank")
                    }
                    truncateTweet = (e, o = 140) => {
                        if (!e) return null;
                        const i = e.length > o;
                        return `${e.substring(0,o)}${i?"...":""}`
                    };
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(M.s), t.rXU(st), t.rXU(Vt), t.rXU(R.nX), t.rXU(Q))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-order-completion"]
                        ],
                        decls: 47,
                        vars: 19,
                        consts: [
                            ["appearance", "outlined", 1, "mat-elevation-z6", "mat-own-card"],
                            [1, "mdc-card"],
                            [1, "order-completion-header"],
                            ["translate", "", 1, "confirmation"],
                            ["translate", ""],
                            ["routerLink", "/track-result/new", "translate", "", 3, "queryParams"],
                            [1, "heading-banner"],
                            [1, "heading-text"],
                            [1, "order-links"],
                            ["mat-icon-button", "", "aria-label", "Tweet", 3, "href"],
                            [1, "fab", "fa-twitter", "fa-lg", "tweet-logo"],
                            ["mat-icon-button", "", "aria-label", "Print order confirmation", "matTooltipPosition", "below", 3, "click", "matTooltip"],
                            [1, "table-container"],
                            [3, "dataSource"],
                            ["matColumnDef", "product"],
                            ["translate", "", 4, "matHeaderCellDef"],
                            [4, "matCellDef"],
                            [4, "matFooterCellDef"],
                            ["matColumnDef", "price"],
                            ["matColumnDef", "quantity"],
                            ["matColumnDef", "total price"],
                            ["class", "price-align", 4, "matCellDef"],
                            [4, "matHeaderRowDef"],
                            [4, "matRowDef", "matRowDefColumns"],
                            ["mat-footer-row", "", 4, "matFooterRowDef"],
                            [1, "bonus-point-container"],
                            ["translate", "", 3, "translateParams"],
                            ["translate", "", 1, "confirmation", 3, "translateParams"],
                            [1, "mat-table"],
                            [1, "mat-row"],
                            [1, "mat-row", 2, "font-weight", "bold"],
                            [1, "price-align"],
                            ["mat-footer-row", ""]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "mat-card", 0)(1, "div", 1)(2, "div", 2)(3, "div")(4, "h1", 3), t.EFF(5, "THANKS_FOR_PURCHASE"), t.k0s(), t.j41(6, "div", 4), t.EFF(7, "PURCHASE_COMMENT_PREFIX "), t.j41(8, "a", 5), t.EFF(9, "TITLE_TRACK_ORDERS"), t.k0s(), t.EFF(10, " PURCHASE_COMMENT_SUFFIX"), t.k0s()(), t.nVh(11, Vl, 15, 8, "div"), t.k0s(), t.j41(12, "div", 6)(13, "span", 7), t.EFF(14), t.nI1(15, "translate"), t.k0s(), t.j41(16, "div", 8)(17, "a", 9), t.nrm(18, "i", 10), t.k0s(), t.j41(19, "button", 11), t.nI1(20, "translate"), t.bIt("click", function() {
                                return i.openConfirmationPDF()
                            }), t.j41(21, "mat-icon"), t.EFF(22, "note"), t.k0s()()()(), t.j41(23, "div", 12)(24, "mat-table", 13), t.qex(25, 14), t.DNE(26, Gl, 2, 0, "mat-header-cell", 15)(27, Yl, 2, 1, "mat-cell", 16)(28, Hl, 1, 0, "mat-footer-cell", 17), t.bVm(), t.qex(29, 18), t.DNE(30, Xl, 2, 0, "mat-header-cell", 15)(31, Wl, 2, 1, "mat-cell", 16)(32, zl, 1, 0, "mat-footer-cell", 17), t.bVm(), t.qex(33, 19), t.DNE(34, Jl, 2, 0, "mat-header-cell", 15)(35, Ql, 3, 1, "mat-cell", 16)(36, ql, 14, 0, "mat-footer-cell", 17), t.bVm(), t.qex(37, 20), t.DNE(38, Kl, 2, 0, "mat-header-cell", 15)(39, Zl, 2, 1, "mat-cell", 21)(40, tc, 14, 4, "mat-footer-cell", 17), t.bVm(), t.DNE(41, ec, 1, 0, "mat-header-row", 22)(42, nc, 1, 0, "mat-row", 23)(43, oc, 1, 0, "mat-footer-row", 24), t.k0s()(), t.j41(44, "div", 25)(45, "span", 26), t.EFF(46, "BONUS_POINTS_COUNT"), t.k0s()()()()), 2 & o && (t.R7$(8), t.Y8G("queryParams", t.eq3(15, Ll, i.orderId)), t.R7$(3), t.vxM(i.address ? 11 : -1), t.R7$(3), t.JRh(t.bMT(15, 11, "ORDER_SUMMARY")), t.R7$(3), t.Y8G("href", t.VkB("https://twitter.com/intent/tweet?text=", i.tweetText, "&hashtags=security"), t.B4B), t.R7$(2), t.Y8G("matTooltip", t.bMT(20, 13, "PRINT_ORDER_CONFIRMATION")), t.R7$(5), t.Y8G("dataSource", i.dataSource), t.R7$(17), t.Y8G("matHeaderRowDef", i.tableColumns), t.R7$(), t.Y8G("matRowDefColumns", i.tableColumns), t.R7$(), t.Y8G("matFooterRowDef", i.tableColumns), t.R7$(2), t.Y8G("translateParams", t.eq3(17, Bl, i.orderDetails.bonus)))
                        },
                        dependencies: [h.Hu, h.RN, d.h, d.Mm, nt.Wk, V.iY, B.oV, f.m_, f.An, u.Zl, u.cC, u.tL, u.KS, u.YV, u.$R, u.Zq, u.Qo, u.ji, u.YZ, u.iL, u.NB, u.xW, u.iF, d.D9],
                        styles: ["\n\n\n\n.order-completion-header[_ngcontent-%COMP%]{display:grid;gap:24px;grid-template-columns:auto minmax(250px,max-content);margin-bottom:16px}@media (max-width: 800px){.order-completion-header[_ngcontent-%COMP%]{grid-template-columns:auto}}.order-completion-header[_ngcontent-%COMP%] > div[_ngcontent-%COMP%]{padding:8px}.heading-banner[_ngcontent-%COMP%]{align-items:center;background:#0003;display:flex;font-size:x-large;justify-content:space-between;padding:6px 8px 6px 20px}.order-links[_ngcontent-%COMP%]{align-items:center;display:flex}.heading-text[_ngcontent-%COMP%]{justify-content:center}.mat-own-card[_ngcontent-%COMP%]{display:block;margin-left:20%;margin-right:20%}mat-cell[_ngcontent-%COMP%], mat-footer-cell[_ngcontent-%COMP%], mat-header-cell[_ngcontent-%COMP%]{padding-left:10px;padding-right:10px}.price-align[_ngcontent-%COMP%]{text-align:right}.tweet-logo[_ngcontent-%COMP%]{height:24px}.table-container[_ngcontent-%COMP%]{--mat-table-footer-container-height: auto}.bonus-point-container[_ngcontent-%COMP%]{margin-top:16px}"]
                    })
                }
                return n
            })();
            const ac = n => ({
                bonus: n
            });

            function rc(n, a) {
                if (1 & n && (t.j41(0, "mat-card", 18)(1, "div", 7)(2, "div")(3, "b"), t.EFF(4), t.nI1(5, "translate"), t.k0s()(), t.j41(6, "div")(7, "span"), t.EFF(8), t.nI1(9, "translate"), t.k0s(), t.EFF(10), t.k0s(), t.j41(11, "div")(12, "span"), t.EFF(13), t.nI1(14, "translate"), t.k0s(), t.EFF(15), t.k0s()()()), 2 & n) {
                    const e = t.XpG(2);
                    t.R7$(4), t.JRh(t.bMT(5, 5, "PAYMENT_METHOD")), t.R7$(4), t.JRh(t.bMT(9, 7, "CARD_ENDING_IN")), t.R7$(2), t.SpI(" ", null == e.paymentMethod ? null : e.paymentMethod.cardNum), t.R7$(3), t.SpI("", t.bMT(14, 9, "CARD_HOLDER"), '"'), t.R7$(2), t.SpI(" ", null == e.paymentMethod ? null : e.paymentMethod.fullName)
                }
            }

            function sc(n, a) {
                1 & n && (t.j41(0, "mat-card", 18)(1, "div", 7)(2, "div")(3, "b"), t.EFF(4), t.nI1(5, "translate"), t.k0s()(), t.j41(6, "div")(7, "span"), t.EFF(8), t.nI1(9, "translate"), t.k0s()()()()), 2 & n && (t.R7$(4), t.JRh(t.bMT(5, 2, "PAYMENT_METHOD")), t.R7$(4), t.JRh(t.bMT(9, 4, "DIGITAL_WALLET")))
            }

            function lc(n, a) {
                if (1 & n && (t.j41(0, "div", 3)(1, "mat-card", 17)(2, "div", 7)(3, "div")(4, "b"), t.EFF(5), t.nI1(6, "translate"), t.k0s()(), t.j41(7, "div"), t.EFF(8), t.k0s(), t.j41(9, "div"), t.EFF(10), t.k0s(), t.j41(11, "div"), t.EFF(12), t.k0s(), t.j41(13, "div")(14, "span"), t.EFF(15), t.nI1(16, "translate"), t.k0s(), t.EFF(17), t.k0s()()(), t.nVh(18, rc, 16, 11, "mat-card", 18)(19, sc, 10, 6, "mat-card", 18), t.k0s()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(5), t.JRh(t.bMT(6, 10, "LABEL_DELIVERY_ADDRESS")), t.R7$(3), t.JRh(null == e.address ? null : e.address.fullName), t.R7$(2), t.LHq("", null == e.address ? null : e.address.streetAddress, ", ", null == e.address ? null : e.address.city, ", ", null == e.address ? null : e.address.state, ", ", null == e.address ? null : e.address.zipCode), t.R7$(2), t.JRh(null == e.address ? null : e.address.country), t.R7$(3), t.JRh(t.bMT(16, 12, "PHONE_NUMBER")), t.R7$(2), t.SpI(" ", null == e.address ? null : e.address.mobileNum), t.R7$(), t.vxM("wallet" !== e.paymentMethod ? 18 : 19)
                }
            }
            let cc = (() => {
                class n {
                    router;
                    addressService;
                    paymentService;
                    basketService;
                    deliveryService;
                    ngZone;
                    snackBarHelperService;
                    bonus = 0;
                    itemTotal = 0;
                    deliveryPrice = 0;
                    promotionalDiscount = 0;
                    address;
                    paymentMethod;
                    constructor(e, o, i, r, c, p, k) {
                        this.router = e, this.addressService = o, this.paymentService = i, this.basketService = r, this.deliveryService = c, this.ngZone = p, this.snackBarHelperService = k
                    }
                    ngOnInit() {
                        this.deliveryService.getById(sessionStorage.getItem("deliveryMethodId")).subscribe(e => {
                            this.deliveryPrice = e.price
                        }), this.addressService.getById(sessionStorage.getItem("addressId")).subscribe({
                            next: e => {
                                this.address = e
                            },
                            error: e => {
                                console.log(e)
                            }
                        }), "wallet" !== sessionStorage.getItem("paymentId") ? this.paymentService.getById(sessionStorage.getItem("paymentId")).subscribe({
                            next: e => {
                                e.cardNum = String(e.cardNum).substring(String(e.cardNum).length - 4), this.paymentMethod = e
                            },
                            error: e => {
                                console.log(e)
                            }
                        }) : "wallet" === sessionStorage.getItem("paymentId") && (this.paymentMethod = "wallet")
                    }
                    getMessage(e) {
                        this.itemTotal = e[0], this.promotionalDiscount = sessionStorage.getItem("couponDiscount") ? parseFloat(sessionStorage.getItem("couponDiscount")) / 100 * this.itemTotal : 0, this.bonus = e[1]
                    }
                    placeOrder() {
                        var e = this;
                        const o = {
                            paymentId: sessionStorage.getItem("paymentId"),
                            addressId: sessionStorage.getItem("addressId"),
                            deliveryMethodId: sessionStorage.getItem("deliveryMethodId")
                        };
                        this.basketService.checkout(Number(sessionStorage.getItem("bid")), btoa(sessionStorage.getItem("couponDetails")), o).subscribe({
                            next: i => {
                                sessionStorage.removeItem("paymentId"), sessionStorage.removeItem("addressId"), sessionStorage.removeItem("deliveryMethodId"), sessionStorage.removeItem("couponDetails"), sessionStorage.removeItem("couponDiscount"), this.basketService.updateNumberOfCartItems(), this.ngZone.run((0, F.A)(function*() {
                                    return yield e.router.navigate(["/order-completion", i])
                                }))
                            },
                            error: i => {
                                console.log(i), this.snackBarHelperService.open(i.error?.error.message, "errorBar")
                            }
                        })
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(R.Ix), t.rXU(st), t.rXU(qt), t.rXU(Q), t.rXU(Et), t.rXU(t.SKi), t.rXU(L.r))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-order-summary"]
                        ],
                        decls: 46,
                        vars: 34,
                        consts: [
                            ["appearance", "outlined", 1, "container", "mat-elevation-z6", "mat-own-card"],
                            [1, "mdc-card", "container"],
                            [1, "column", "flex-50"],
                            [1, "container"],
                            ["id", "app-purchase-basket", 3, "emitTotal", "allowEdit", "totalPrice"],
                            [1, "flex-50"],
                            ["appearance", "outlined", 1, "mat-elevation-z0", "fill"],
                            [1, "mdc-card"],
                            [1, "order-summary"],
                            [1, "mat-table"],
                            [1, "mat-row"],
                            [1, "mat-cell", "label"],
                            [1, "mat-cell", "price"],
                            [1, "mat-footer-cell", "label"],
                            [1, "mat-footer-cell", "price"],
                            ["mat-raised-button", "", "mat-button", "", "color", "primary", "aria-label", "Complete your purchase", "id", "checkoutButton", 1, "btn", "btn-pay", 3, "click"],
                            [1, "bonus-points"],
                            ["appearance", "outlined", 1, "mat-elevation-z0", "flex-60"],
                            ["appearance", "outlined", 1, "mat-elevation-z0", "flex-40"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "mat-card", 0)(1, "div", 1)(2, "div", 2), t.nVh(3, lc, 20, 14, "div", 3), t.j41(4, "app-purchase-basket", 4), t.bIt("emitTotal", function(c) {
                                return i.getMessage(c)
                            }), t.k0s()(), t.j41(5, "div", 5)(6, "mat-card", 6)(7, "div", 7)(8, "div", 8), t.EFF(9), t.nI1(10, "translate"), t.k0s(), t.j41(11, "table", 9)(12, "tr", 10)(13, "td", 11), t.EFF(14), t.nI1(15, "translate"), t.k0s(), t.j41(16, "td", 12), t.EFF(17), t.k0s()(), t.j41(18, "tr", 10)(19, "td", 11), t.EFF(20), t.nI1(21, "translate"), t.k0s(), t.j41(22, "td", 12), t.EFF(23), t.k0s()(), t.j41(24, "tr", 10)(25, "td", 11), t.EFF(26), t.nI1(27, "translate"), t.k0s(), t.j41(28, "td", 12), t.EFF(29), t.k0s()(), t.j41(30, "tr", 10)(31, "td", 13), t.EFF(32), t.nI1(33, "translate"), t.k0s(), t.j41(34, "td", 14), t.EFF(35), t.k0s()()(), t.j41(36, "button", 15), t.bIt("click", function() {
                                return i.placeOrder()
                            }), t.j41(37, "mat-icon"), t.EFF(38), t.nI1(39, "translate"), t.k0s(), t.j41(40, "span"), t.EFF(41), t.nI1(42, "translate"), t.k0s()(), t.j41(43, "div", 16), t.EFF(44), t.nI1(45, "translate"), t.k0s()()()()()()), 2 & o && (t.R7$(3), t.vxM(i.address && i.paymentMethod ? 3 : -1), t.R7$(), t.Y8G("allowEdit", !1)("totalPrice", !1), t.R7$(5), t.JRh(t.bMT(10, 15, "ORDER_SUMMARY")), t.R7$(5), t.JRh(t.bMT(15, 17, "ITEMS")), t.R7$(3), t.SpI("", null == i.itemTotal ? null : i.itemTotal.toFixed(2), "\xa4"), t.R7$(3), t.JRh(t.bMT(21, 19, "DELIVERY")), t.R7$(3), t.SpI("", i.deliveryPrice.toFixed(2), "\xa4"), t.R7$(3), t.JRh(t.bMT(27, 21, "PROMOTION")), t.R7$(3), t.SpI("", i.promotionalDiscount.toFixed(2), "\xa4"), t.R7$(3), t.JRh(t.bMT(33, 23, "LABEL_TOTAL_PRICE")), t.R7$(3), t.SpI("", i.itemTotal + i.deliveryPrice - i.promotionalDiscount == null ? null : (i.itemTotal + i.deliveryPrice - i.promotionalDiscount).toFixed(2), "\xa4"), t.R7$(3), t.JRh(t.bMT(39, 25, "monetization_on")), t.R7$(3), t.JRh(t.bMT(42, 27, "PLACE_ORDER_AND_PAY")), t.R7$(3), t.JRh(t.i5U(45, 29, "CHECKOUT_FOR_BONUS_POINTS", t.eq3(32, ac, i.bonus))))
                        },
                        dependencies: [h.Hu, h.RN, d.h, ce, b.Hl, b.$z, f.m_, f.An, d.D9],
                        styles: ["\n\n\n\n.btn-pay[_ngcontent-%COMP%]{display:block;margin-top:20px;padding:10px;width:100%}mat-card[_ngcontent-%COMP%]{height:auto;margin-bottom:-10px!important;margin-top:20px}mat-cell[_ngcontent-%COMP%]{padding-left:10px;padding-right:10px}.price[_ngcontent-%COMP%]:last-of-type{padding-right:10px;text-align:right}.label[_ngcontent-%COMP%]:first-of-type{padding-left:10px}table[_ngcontent-%COMP%]{width:100%}.mat-mdc-footer-cell[_ngcontent-%COMP%]{font-weight:700}.order-summary[_ngcontent-%COMP%]{font-size:x-large;font-weight:500;margin-bottom:15px;padding-bottom:10px;padding-left:10px}.bonus-points[_ngcontent-%COMP%]{color:var(--theme-text);margin-top:10px;text-align:center}.mdc-card[_ngcontent-%COMP%]{border:0;border-bottom:0;margin-top:-5px;padding:5px;width:100%}.mat-own-card[_ngcontent-%COMP%]{background-color:var(--theme-background);margin:auto;max-width:1200px;padding:20px;width:60%!important}.mat-table[_ngcontent-%COMP%]   .mat-row[_ngcontent-%COMP%], .mat-table[_ngcontent-%COMP%]   .mat-footer-cell[_ngcontent-%COMP%]{border-bottom:1px solid var(--theme-background-light);border-radius:5px!important;padding:10px 0}.mat-table[_ngcontent-%COMP%]{background:transparent;border-collapse:collapse;width:100%}.mat-cell[_ngcontent-%COMP%], .mat-footer-cell[_ngcontent-%COMP%]{color:var(--theme-text);padding:12px 0}.mat-footer-cell.label[_ngcontent-%COMP%]{font-weight:700;text-align:start;text-indent:10px}.mat-footer-cell.price[_ngcontent-%COMP%]{font-weight:700;padding-right:10px!important;text-align:end}.container[_ngcontent-%COMP%]{align-items:stretch;display:flex;flex-direction:row;gap:20px}.column[_ngcontent-%COMP%]{display:flex;flex-direction:column;gap:20px}.flex-50[_ngcontent-%COMP%]{flex:0 0 50%;max-width:50%;min-width:0}.flex-60[_ngcontent-%COMP%]{flex:0 0 60%;max-width:60%;min-width:0}.flex-40[_ngcontent-%COMP%]{flex:0 0 40%;max-width:40%;min-width:0}.fill[_ngcontent-%COMP%]{height:auto;width:100%}@media (max-width: 959.98px){.container[_ngcontent-%COMP%]{flex-direction:column}.flex-50[_ngcontent-%COMP%], .flex-60[_ngcontent-%COMP%], .flex-40[_ngcontent-%COMP%]{flex:0 0 100%;max-width:100%}}"]
                    })
                }
                return n
            })();

            function dc(n, a) {
                1 & n && (t.j41(0, "mat-error"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.SpI(" ", t.bMT(2, 1, "MANDATORY_AMOUNT"), " "))
            }

            function mc(n, a) {
                1 & n && (t.j41(0, "mat-error"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.SpI(" ", t.bMT(2, 1, "AMOUNT_LIMIT"), " "))
            }
            let pc = (() => {
                class n {
                    router;
                    walletService;
                    ngZone;
                    balance;
                    balanceControl = new l.hs("", [l.k0.required, l.k0.min(10), l.k0.max(1e3)]);
                    constructor(e, o, i) {
                        this.router = e, this.walletService = o, this.ngZone = i
                    }
                    ngOnInit() {
                        this.walletService.get().subscribe({
                            next: e => {
                                this.balance = parseFloat(e).toFixed(2)
                            },
                            error: e => {
                                console.log(e)
                            }
                        })
                    }
                    continue () {
                        var e = this;
                        sessionStorage.setItem("walletTotal", this.balanceControl.value), this.ngZone.run((0, F.A)(function*() {
                            return yield e.router.navigate(["/payment", "wallet"])
                        }))
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(R.Ix), t.rXU(Qt), t.rXU(t.SKi))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-wallet"]
                        ],
                        decls: 31,
                        vars: 23,
                        consts: [
                            ["appearance", "outlined", 1, "mat-elevation-z6"],
                            [1, "mdc-card"],
                            [1, "header_container"],
                            ["href", "/#/wallet-web3"],
                            ["translate", ""],
                            [1, "confirmation"],
                            ["appearance", "outline", "color", "accent"],
                            ["type", "number", "matInput", "", "aria-label", "Enter an amount", 3, "formControl"],
                            ["type", "submit", "id", "submitButton", "mat-raised-button", "", "color", "primary", "aria-label", "Button to continue to payment", 3, "click", "disabled"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "mat-card", 0)(1, "div", 1)(2, "div", 2)(3, "h1"), t.EFF(4), t.nI1(5, "translate"), t.k0s(), t.j41(6, "a", 3), t.EFF(7), t.nI1(8, "translate"), t.k0s()(), t.j41(9, "h3"), t.EFF(10), t.nI1(11, "translate"), t.k0s(), t.j41(12, "p")(13, "b")(14, "span", 4), t.EFF(15), t.nI1(16, "translate"), t.k0s(), t.j41(17, "span", 5), t.EFF(18), t.k0s()()(), t.j41(19, "mat-form-field", 6)(20, "mat-label"), t.EFF(21), t.nI1(22, "translate"), t.k0s(), t.nrm(23, "input", 7), t.nVh(24, dc, 3, 3, "mat-error"), t.nVh(25, mc, 3, 3, "mat-error"), t.k0s(), t.j41(26, "button", 8), t.bIt("click", function() {
                                return i.continue()
                            }), t.j41(27, "mat-icon"), t.EFF(28, " monetization_on "), t.k0s(), t.EFF(29), t.nI1(30, "translate"), t.k0s()()()), 2 & o && (t.R7$(4), t.JRh(t.bMT(5, 11, "DIGITAL_WALLET")), t.R7$(3), t.JRh(t.bMT(8, 13, "LINK_TRY_OUT_NEW_CRYPTO_WALLET")), t.R7$(3), t.JRh(t.bMT(11, 15, "LABEL_ADD_MONEY")), t.R7$(5), t.JRh(t.bMT(16, 17, "LABEL_WALLET_BALANCE")), t.R7$(3), t.SpI(" ", i.balance), t.R7$(3), t.JRh(t.bMT(22, 19, "LABEL_AMOUNT")), t.R7$(2), t.Y8G("formControl", i.balanceControl), t.R7$(), t.vxM(i.balanceControl.invalid && i.balanceControl.errors.required ? 24 : -1), t.R7$(), t.vxM(i.balanceControl.invalid && (i.balanceControl.errors.min || i.balanceControl.errors.max) ? 25 : -1), t.R7$(), t.Y8G("disabled", i.balanceControl.invalid), t.R7$(3), t.SpI(" ", t.bMT(30, 21, "BTN_DEPOSIT"), " "))
                        },
                        dependencies: [h.Hu, h.RN, d.h, d.Mm, I.R, _.rl, _.nJ, _.TL, l.YN, l.me, l.Q0, l.BC, T.fS, T.fg, l.X1, l.l_, b.Hl, b.$z, f.m_, f.An, d.D9],
                        styles: ["\n\n\n\nmat-card[_ngcontent-%COMP%]{margin-left:30%;margin-right:30%}mat-form-field[_ngcontent-%COMP%]{padding-top:10px;width:100%}.form-container[_ngcontent-%COMP%]{min-width:350px}#submitButton[_ngcontent-%COMP%]{margin-left:20%;margin-top:30px;width:60%}.heading[_ngcontent-%COMP%]{background:#0003;font-size:x-large;justify-content:center;padding:12px 20px}.header_container[_ngcontent-%COMP%]{display:flex;justify-content:space-between}.mdc-card[_ngcontent-%COMP%]{border:0}@media (max-width: 1100px){mat-card[_ngcontent-%COMP%]{margin-left:20%;margin-right:20%}}@media (max-width: 580px){mat-card[_ngcontent-%COMP%]{margin-left:10%;margin-right:10%}.header_container[_ngcontent-%COMP%]{flex-direction:column;margin-bottom:16px}}"]
                    })
                }
                return n
            })();

            function uc(n, a) {
                1 & n && (t.j41(0, "div", 10), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.JRh(t.bMT(2, 1, "LABEL_IN_TRANSIT")))
            }

            function hc(n, a) {
                1 & n && (t.j41(0, "div", 11), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.JRh(t.bMT(2, 1, "LABEL_DELIVERED")))
            }

            function gc(n, a) {
                1 & n && (t.j41(0, "mat-header-cell"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.JRh(t.bMT(2, 1, "LABEL_PRODUCT")))
            }

            function fc(n, a) {
                if (1 & n && (t.j41(0, "mat-cell"), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.JRh(e.name)
                }
            }

            function _c(n, a) {
                1 & n && (t.j41(0, "mat-header-cell"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.JRh(t.bMT(2, 1, "LABEL_PRICE")))
            }

            function Cc(n, a) {
                if (1 & n && (t.j41(0, "mat-cell"), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.SpI("", e.price, "\xa4")
                }
            }

            function bc(n, a) {
                1 & n && (t.j41(0, "mat-header-cell"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.JRh(t.bMT(2, 1, "LABEL_QUANTITY")))
            }

            function vc(n, a) {
                if (1 & n && (t.j41(0, "mat-cell")(1, "span"), t.EFF(2), t.k0s()()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(2), t.JRh(e.quantity)
                }
            }

            function kc(n, a) {
                1 & n && (t.j41(0, "mat-header-cell"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.JRh(t.bMT(2, 1, "LABEL_TOTAL_PRICE")))
            }

            function Sc(n, a) {
                if (1 & n && (t.j41(0, "mat-cell", 27), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.SpI("", e.total.toFixed(2), "\xa4")
                }
            }

            function Fc(n, a) {
                1 & n && t.nrm(0, "mat-header-cell")
            }

            function xc(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-cell")(1, "button", 14), t.nI1(2, "translate"), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG(3);
                        return s.Njj(r.showDetail(i.id))
                    }), t.j41(3, "mat-icon"), t.EFF(4, "rate_review"), t.k0s()()()
                }
                2 & n && (t.R7$(), t.Y8G("matTooltip", t.mNQ(t.bMT(2, 2, "WRITE_REVIEW"))))
            }

            function Tc(n, a) {
                1 & n && t.nrm(0, "mat-header-row")
            }

            function yc(n, a) {
                1 & n && t.nrm(0, "mat-row")
            }

            function Rc(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "div")(1, "div", 4)(2, "div", 5)(3, "div", 6)(4, "div", 7)(5, "div"), t.EFF(6), t.nI1(7, "translate"), t.k0s(), t.j41(8, "div"), t.EFF(9), t.k0s()(), t.j41(10, "div", 8)(11, "div"), t.EFF(12), t.nI1(13, "translate"), t.k0s(), t.j41(14, "div"), t.EFF(15), t.k0s()(), t.j41(16, "div", 8)(17, "div"), t.EFF(18), t.nI1(19, "translate"), t.k0s(), t.j41(20, "div"), t.EFF(21), t.k0s()(), t.j41(22, "div", 9), t.nVh(23, uc, 3, 3, "div", 10), t.nVh(24, hc, 3, 3, "div", 11), t.k0s(), t.j41(25, "div", 12)(26, "button", 13), t.nI1(27, "translate"), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG(2);
                        return s.Njj(r.trackOrder(i.orderId))
                    }), t.j41(28, "mat-icon"), t.EFF(29, "local_shipping"), t.k0s()()(), t.j41(30, "div", 12)(31, "button", 14), t.nI1(32, "translate"), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG(2);
                        return s.Njj(r.openConfirmationPDF(i.orderId))
                    }), t.j41(33, "mat-icon"), t.EFF(34, "note"), t.k0s()()()(), t.nrm(35, "div", 15), t.k0s(), t.j41(36, "mat-table", 16), t.qex(37, 17), t.DNE(38, gc, 3, 3, "mat-header-cell", 18)(39, fc, 2, 1, "mat-cell", 19), t.bVm(), t.qex(40, 20), t.DNE(41, _c, 3, 3, "mat-header-cell", 18)(42, Cc, 2, 1, "mat-cell", 19), t.bVm(), t.qex(43, 21), t.DNE(44, bc, 3, 3, "mat-header-cell", 18)(45, vc, 3, 1, "mat-cell", 19), t.bVm(), t.qex(46, 22), t.DNE(47, kc, 3, 3, "mat-header-cell", 18)(48, Sc, 2, 1, "mat-cell", 23), t.bVm(), t.qex(49, 24), t.DNE(50, Fc, 1, 0, "mat-header-cell", 18)(51, xc, 5, 4, "mat-cell", 19), t.bVm(), t.DNE(52, Tc, 1, 0, "mat-header-row", 25)(53, yc, 1, 0, "mat-row", 26), t.k0s()()()
                }
                if (2 & n) {
                    const e = a.$implicit,
                        o = t.XpG(2);
                    t.R7$(6), t.JRh(t.bMT(7, 15, "LABEL_ORDER_ID")), t.R7$(3), t.SpI("#", e.orderId), t.R7$(3), t.JRh(t.bMT(13, 17, "LABEL_TOTAL_PRICE")), t.R7$(3), t.SpI("", e.totalPrice.toFixed(2), "\xa4"), t.R7$(3), t.JRh(t.bMT(19, 19, "LABEL_BONUS")), t.R7$(3), t.JRh(e.bonus), t.R7$(2), t.vxM(e.delivered ? -1 : 23), t.R7$(), t.vxM(e.delivered ? 24 : -1), t.R7$(2), t.Y8G("matTooltip", t.mNQ(t.bMT(27, 21, "LABEL_TRACK_ORDER"))), t.R7$(5), t.Y8G("matTooltip", t.mNQ(t.bMT(32, 23, "PRINT_ORDER_CONFIRMATION"))), t.R7$(5), t.Y8G("dataSource", e.products), t.R7$(16), t.Y8G("matHeaderRowDef", o.tableColumns), t.R7$(), t.Y8G("matRowDefColumns", o.tableColumns)
                }
            }

            function wc(n, a) {
                if (1 & n && (t.j41(0, "div", 2), t.Z7z(1, Rc, 54, 25, "div", null, t.fX1), t.k0s()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(), t.Dyx(e.orders)
                }
            }

            function Ec(n, a) {
                1 & n && (t.j41(0, "mat-card", 3)(1, "div", 28), t.nrm(2, "img", 29), t.j41(3, "mat-card-title")(4, "span", 30), t.EFF(5), t.nI1(6, "translate"), t.k0s()(), t.j41(7, "mat-card-content")(8, "span", 31), t.EFF(9), t.nI1(10, "translate"), t.k0s()()()()), 2 & n && (t.R7$(5), t.SpI(" ", t.bMT(6, 2, "NO_SEARCH_RESULT"), " "), t.R7$(4), t.SpI(" ", t.bMT(10, 4, "NO_ORDERS_PLACED"), " "))
            }
            let Mc = (() => {
                class n {
                    router;
                    dialog;
                    orderHistoryService;
                    basketService;
                    productService;
                    ngZone;
                    tableColumns = ["product", "price", "quantity", "total price", "review"];
                    orders = [];
                    emptyState = !0;
                    constructor(e, o, i, r, c, p) {
                        this.router = e, this.dialog = o, this.orderHistoryService = i, this.basketService = r, this.productService = c, this.ngZone = p
                    }
                    ngOnInit() {
                        this.orderHistoryService.get().subscribe({
                            next: e => {
                                e = e.reverse(), this.emptyState = 0 === e.length;
                                for (const o of e) {
                                    const i = [];
                                    for (const r of o.products) i.push({
                                        id: r.id,
                                        name: r.name,
                                        price: r.price,
                                        quantity: r.quantity,
                                        total: r.total
                                    });
                                    this.orders.push({
                                        orderId: o.orderId,
                                        totalPrice: o.totalPrice,
                                        bonus: o.bonus,
                                        products: new u.I6(i),
                                        delivered: o.delivered
                                    })
                                }
                            },
                            error: e => {
                                console.log(e)
                            }
                        })
                    }
                    showDetail(e) {
                        this.productService.get(e).subscribe({
                            next: o => {
                                const i = {
                                    id: o.id,
                                    name: o.name,
                                    description: o.description,
                                    image: o.image,
                                    price: o.price,
                                    points: Math.round(o.price / 10)
                                };
                                this.dialog.open(Ce, {
                                    width: "500px",
                                    height: "max-content",
                                    data: {
                                        productData: i
                                    }
                                })
                            },
                            error: o => {
                                console.log(o)
                            }
                        })
                    }
                    openConfirmationPDF(e) {
                        window.open(`${this.basketService.hostServer}/ftp/order_${e}.pdf`, "_blank")
                    }
                    trackOrder(e) {
                        var o = this;
                        this.ngZone.run((0, F.A)(function*() {
                            return yield o.router.navigate(["/track-result"], {
                                queryParams: {
                                    id: e
                                }
                            })
                        }))
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(R.Ix), t.rXU(w.bZ), t.rXU(Kt), t.rXU(Q), t.rXU(Rt), t.rXU(t.SKi))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-order-history"]
                        ],
                        decls: 7,
                        vars: 4,
                        consts: [
                            ["appearance", "outlined", 1, "card1", "mat-elevation-z6"],
                            [1, "mdc-card", "mat-card-cvr"],
                            [1, "orders-container"],
                            ["appearance", "outlined", 1, "mat-elevation-z6", "emptyState"],
                            [1, "table-container", "mat-elevation-z4", "custom-slate"],
                            [1, "heading"],
                            [1, "heading-row"],
                            [1, "col", "col-40", "col-vertical"],
                            [1, "col", "col-15", "col-vertical"],
                            [1, "col", "col-20"],
                            [1, "error"],
                            [1, "confirmation"],
                            [1, "col", "col-5", "align-end"],
                            ["mat-icon-button", "", "aria-label", "Track Your Order", "matTooltipPosition", "below", 3, "click", "matTooltip"],
                            ["mat-icon-button", "", "aria-label", "Print order confirmation", "matTooltipPosition", "below", 3, "click", "matTooltip"],
                            [1, "border"],
                            [3, "dataSource"],
                            ["matColumnDef", "product"],
                            [4, "matHeaderCellDef"],
                            [4, "matCellDef"],
                            ["matColumnDef", "price"],
                            ["matColumnDef", "quantity"],
                            ["matColumnDef", "total price"],
                            ["class", "price-align", 4, "matCellDef"],
                            ["matColumnDef", "review"],
                            [4, "matHeaderRowDef"],
                            [4, "matRowDef", "matRowDefColumns"],
                            [1, "price-align"],
                            [1, "mdc-card"],
                            ["alt", "No results found", "src", "assets/public/images/products/no-results.png", 1, "img-responsive", "noResult"],
                            [1, "noResultText"],
                            ["translate", "", 1, "noResultText"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "mat-card", 0)(1, "div", 1)(2, "mat-card-title"), t.EFF(3), t.nI1(4, "translate"), t.k0s(), t.nVh(5, wc, 3, 0, "div", 2)(6, Ec, 11, 6, "mat-card", 3), t.k0s()()), 2 & o && (t.R7$(3), t.SpI(" ", t.bMT(4, 2, "LABEL_ORDER_HISTORY"), " "), t.R7$(2), t.vxM(i.emptyState ? 6 : 5))
                        },
                        dependencies: [h.Hu, h.RN, h.m2, h.dh, d.h, d.Mm, V.iY, B.oV, f.m_, f.An, u.Zl, u.cC, u.tL, u.KS, u.YV, u.$R, u.ji, u.YZ, u.iL, u.NB, d.D9],
                        styles: ["\n\n\n\n.heading[_ngcontent-%COMP%]{padding:12px 20px}.heading-text[_ngcontent-%COMP%]{justify-content:center;padding:12px 20px}mat-card[_ngcontent-%COMP%]{height:auto;width:80%}.card1[_ngcontent-%COMP%]{border-radius:5px;margin:0 auto;width:80%}.main-wrapper[_ngcontent-%COMP%]{margin-left:10%;margin-right:10%}mat-cell[_ngcontent-%COMP%], mat-footer-cell[_ngcontent-%COMP%]{padding-left:10px;padding-right:10px}.price-align[_ngcontent-%COMP%]{text-align:right}mat-card-title[_ngcontent-%COMP%]{font-size:24px;margin-bottom:25px}.border[_ngcontent-%COMP%]{border:.5px solid #f2f2f2}.emptyState[_ngcontent-%COMP%]{display:block;height:auto!important;margin:20px auto;width:50%}.custom-slate[_ngcontent-%COMP%]{border-radius:5px}.noResult[_ngcontent-%COMP%]{display:block;margin-left:auto;margin-right:auto;width:60%}.noResultText[_ngcontent-%COMP%]{display:block;margin-top:10px;text-align:center}.mat-mdc-header-cell[_ngcontent-%COMP%]:first-child{padding-left:24px!important}.mat-mdc-header-cell[_ngcontent-%COMP%]:not(:first-child){padding-left:2px!important}.mat-mdc-header-cell[_ngcontent-%COMP%]{align-items:center;color:var(--theme-text-darker)!important;display:flex;font-size:12px;justify-content:flex-start}.mdc-card[_ngcontent-%COMP%]{border:0}.orders-container[_ngcontent-%COMP%]{display:flex;flex-direction:column;gap:20px}.heading-row[_ngcontent-%COMP%]{align-items:center;display:flex}.col[_ngcontent-%COMP%]{min-width:0}.col-40[_ngcontent-%COMP%]{flex:0 0 40%;max-width:40%}.col-20[_ngcontent-%COMP%]{flex:0 0 20%;max-width:20%}.col-15[_ngcontent-%COMP%]{flex:0 0 15%;max-width:15%}.col-5[_ngcontent-%COMP%]{flex:0 0 5%;max-width:5%}.col-vertical[_ngcontent-%COMP%]{display:flex;flex-direction:column}.align-end[_ngcontent-%COMP%]{display:flex;justify-content:flex-end}mat-table[_ngcontent-%COMP%]{table-layout:fixed;width:100%}.mat-column-product[_ngcontent-%COMP%]{width:40%}.mat-column-price[_ngcontent-%COMP%], .mat-column-quantity[_ngcontent-%COMP%], .mat-column-total-price[_ngcontent-%COMP%], .mat-column-total\\ price[_ngcontent-%COMP%], .mat-column-totalprice[_ngcontent-%COMP%], .mat-column-total[_ngcontent-%COMP%], .mat-column-review[_ngcontent-%COMP%]{width:15%}"]
                    })
                }
                return n
            })();

            function Ic(n, a) {
                if (1 & n && (t.j41(0, "div", 2)(1, "h1", 5), t.EFF(2, "LABEL_DELIVERY_ADDRESS"), t.k0s(), t.j41(3, "div"), t.EFF(4), t.k0s(), t.j41(5, "div"), t.EFF(6), t.k0s(), t.j41(7, "div"), t.EFF(8), t.k0s(), t.j41(9, "div")(10, "span", 5), t.EFF(11, "PHONE_NUMBER"), t.k0s(), t.EFF(12), t.k0s()()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(4), t.JRh(null == e.address ? null : e.address.fullName), t.R7$(2), t.LHq("", null == e.address ? null : e.address.streetAddress, ", ", null == e.address ? null : e.address.city, ", ", null == e.address ? null : e.address.state, ", ", null == e.address ? null : e.address.zipCode), t.R7$(2), t.JRh(null == e.address ? null : e.address.country), t.R7$(4), t.SpI(" ", null == e.address ? null : e.address.mobileNum)
                }
            }

            function Oc(n, a) {
                1 & n && t.nrm(0, "mat-divider", 3)
            }

            function Pc(n, a) {
                1 & n && t.nrm(0, "mat-header-cell")
            }

            function jc(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-cell")(1, "mat-radio-button", 20), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG();
                        return s.Njj(r.selectMethod(i.id))
                    })("change", function(i) {
                        const r = s.eBV(e).$implicit,
                            c = t.XpG();
                        return s.Njj(i ? c.selection.toggle(r) : null)
                    }), t.k0s()()
                }
                if (2 & n) {
                    const e = a.$implicit,
                        o = t.XpG();
                    t.R7$(), t.Y8G("checked", o.selection.isSelected(e))
                }
            }

            function Ac(n, a) {
                1 & n && t.nrm(0, "mat-header-cell")
            }

            function Dc(n, a) {
                if (1 & n && (t.j41(0, "mat-cell"), t.nrm(1, "i", 21), t.EFF(2), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.Y8G("ngClass", e.icon), t.R7$(), t.SpI("\xa0", e.name)
                }
            }

            function $c(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 22)
            }

            function Lc(n, a) {
                if (1 & n && (t.j41(0, "mat-cell"), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.SpI("", e.price.toFixed(2), "\xa4")
                }
            }

            function Bc(n, a) {
                1 & n && t.nrm(0, "mat-header-cell", 23)
            }

            function Nc(n, a) {
                if (1 & n && (t.j41(0, "mat-cell"), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.Lme("", e.eta, " ", t.bMT(2, 2, "LABEL_DAYS"))
                }
            }

            function Uc(n, a) {
                1 & n && t.nrm(0, "mat-header-row")
            }

            function Vc(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-row", 24), t.bIt("click", function() {
                        const i = s.eBV(e),
                            r = i.$implicit,
                            c = i.$implicit,
                            p = t.XpG();
                        return p.selection.toggle(r), s.Njj(p.selectMethod(c.id))
                    }), t.k0s()
                }
            }
            S.Yv.add(C.KMJ, C.WWd, C.raf);
            let Gc = (() => {
                class n {
                    location;
                    deliverySerivce;
                    addressService;
                    router;
                    ngZone;
                    displayedColumns = ["Selection", "Name", "Price", "ETA"];
                    methods;
                    address;
                    dataSource;
                    deliveryMethodId = void 0;
                    selection = new xe.C(!1, []);
                    constructor(e, o, i, r, c) {
                        this.location = e, this.deliverySerivce = o, this.addressService = i, this.router = r, this.ngZone = c
                    }
                    ngOnInit() {
                        this.addressService.getById(sessionStorage.getItem("addressId")).subscribe({
                            next: e => {
                                this.address = e
                            },
                            error: e => {
                                console.log(e)
                            }
                        }), this.deliverySerivce.get().subscribe({
                            next: e => {
                                console.log(e), this.methods = e, this.dataSource = new u.I6(this.methods)
                            },
                            error: e => {
                                console.log(e)
                            }
                        })
                    }
                    selectMethod(e) {
                        this.deliveryMethodId = this.selection.hasValue() || e ? e : void 0
                    }
                    routeToPreviousUrl() {
                        this.location.back()
                    }
                    chooseDeliveryMethod() {
                        var e = this;
                        sessionStorage.setItem("deliveryMethodId", this.deliveryMethodId.toString()), this.ngZone.run((0, F.A)(function*() {
                            return yield e.router.navigate(["/payment", "shop"])
                        }))
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(zt.aZ), t.rXU(Et), t.rXU(st), t.rXU(R.Ix), t.rXU(t.SKi))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-delivery-method"]
                        ],
                        decls: 34,
                        vars: 9,
                        consts: [
                            ["appearance", "outlined", 1, "mat-elevation-z6"],
                            [1, "mdc-card"],
                            [1, "addressCont"],
                            [1, "detail-divider"],
                            [2, "height", "12px"],
                            ["translate", ""],
                            [3, "dataSource"],
                            ["matColumnDef", "Selection"],
                            [4, "matHeaderCellDef"],
                            [4, "matCellDef"],
                            ["matColumnDef", "Name"],
                            ["matColumnDef", "Price"],
                            ["translate", "LABEL_PRICE", "class", "table-header-label", 4, "matHeaderCellDef"],
                            ["matColumnDef", "ETA"],
                            ["translate", "LABEL_EXPECTED_DELIVERY", "class", "table-header-label", 4, "matHeaderCellDef"],
                            [4, "matHeaderRowDef"],
                            [3, "click", 4, "matRowDef", "matRowDefColumns"],
                            [2, "margin-top", "20px"],
                            ["mat-stroked-button", "", 1, "btn", "btn-return", 3, "click"],
                            ["mat-raised-button", "", "mat-button", "", "color", "primary", "aria-label", "Proceed to delivery method selection", 1, "btn", "nextButton", 3, "click", "disabled"],
                            [3, "click", "change", "checked"],
                            [3, "ngClass"],
                            ["translate", "LABEL_PRICE", 1, "table-header-label"],
                            ["translate", "LABEL_EXPECTED_DELIVERY", 1, "table-header-label"],
                            [3, "click"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "mat-card", 0)(1, "div", 1), t.nVh(2, Ic, 13, 7, "div", 2), t.nVh(3, Oc, 1, 0, "mat-divider", 3), t.nrm(4, "div", 4), t.j41(5, "div")(6, "h1", 5), t.EFF(7, "LABEL_CHOOSE_A_DELIVERY_SPEED"), t.k0s(), t.j41(8, "mat-table", 6), t.qex(9, 7), t.DNE(10, Pc, 1, 0, "mat-header-cell", 8)(11, jc, 2, 1, "mat-cell", 9), t.bVm(), t.qex(12, 10), t.DNE(13, Ac, 1, 0, "mat-header-cell", 8)(14, Dc, 3, 2, "mat-cell", 9), t.bVm(), t.qex(15, 11), t.DNE(16, $c, 1, 0, "mat-header-cell", 12)(17, Lc, 2, 1, "mat-cell", 9), t.bVm(), t.qex(18, 13), t.DNE(19, Bc, 1, 0, "mat-header-cell", 14)(20, Nc, 3, 4, "mat-cell", 9), t.bVm(), t.DNE(21, Uc, 1, 0, "mat-header-row", 15)(22, Vc, 1, 0, "mat-row", 16), t.k0s()(), t.j41(23, "div", 17)(24, "button", 18), t.bIt("click", function() {
                                return i.routeToPreviousUrl()
                            }), t.j41(25, "mat-icon"), t.EFF(26, " navigate_before "), t.k0s(), t.EFF(27), t.nI1(28, "translate"), t.k0s(), t.j41(29, "button", 19), t.bIt("click", function() {
                                return i.chooseDeliveryMethod()
                            }), t.j41(30, "mat-icon"), t.EFF(31, " navigate_next "), t.k0s(), t.j41(32, "span", 5), t.EFF(33, "LABEL_CONTINUE"), t.k0s()()()()()), 2 & o && (t.R7$(2), t.vxM(i.address ? 2 : -1), t.R7$(), t.vxM(i.address ? 3 : -1), t.R7$(5), t.Y8G("dataSource", i.dataSource), t.R7$(13), t.Y8G("matHeaderRowDef", i.displayedColumns), t.R7$(), t.Y8G("matRowDefColumns", i.displayedColumns), t.R7$(5), t.SpI(" ", t.bMT(28, 7, "LABEL_BACK"), " "), t.R7$(2), t.Y8G("disabled", void 0 === i.deliveryMethodId))
                        },
                        dependencies: [h.Hu, h.RN, d.h, d.Mm, U.q, u.Zl, u.cC, u.tL, u.KS, u.YV, u.$R, ot._g, Y.YU, u.ji, u.YZ, u.iL, u.NB, b.Hl, b.$z, f.m_, f.An, d.D9],
                        styles: ["\n\n\n\nmat-card[_ngcontent-%COMP%]{margin-left:20%;margin-right:20%}.addressCont[_ngcontent-%COMP%]{margin-bottom:12px}.detail-divider[_ngcontent-%COMP%]{border-top-width:4px!important;margin-left:1%!important;width:98%!important}.btn-return[_ngcontent-%COMP%]{display:block;margin-left:0;margin-right:auto}.nextButton[_ngcontent-%COMP%]{display:block;margin-left:auto;margin-right:0;margin-top:-37px}.table-header-label[_ngcontent-%COMP%]{color:var(--theme-text-fade-50);font-size:.75rem}mat-table[_ngcontent-%COMP%]{table-layout:fixed;width:100%}.mat-column-Selection[_ngcontent-%COMP%]{width:20%}.mat-column-Name[_ngcontent-%COMP%]{width:30%}.mat-column-Price[_ngcontent-%COMP%], .mat-column-ETA[_ngcontent-%COMP%]{width:25%}"]
                    })
                }
                return n
            })();
            var Mt = m(7673),
                Yc = m(1985);
            const Hc = n => {
                if ("string" == typeof n.value) return (0, Mt.of)(null);
                const a = n.value,
                    e = new FileReader;
                return new Yc.c(i => {
                    e.addEventListener("loadend", () => {
                        const r = new Uint8Array(e.result).subarray(0, 4);
                        let c = "",
                            p = !1;
                        for (let k = 0; k < r.length; k++) c += r[k].toString(16);
                        switch (c) {
                            case "89504e47":
                            case "ffd8ffe0":
                            case "ffd8ffe1":
                            case "ffd8ffe2":
                            case "ffd8ffe3":
                            case "ffd8ffe8":
                                p = !0;
                                break;
                            default:
                                p = !1
                        }
                        i.next(p ? null : {
                            invalidMimeType: !0
                        }), i.complete()
                    }), e.readAsArrayBuffer(a)
                })
            };
            let Oe = (() => {
                class n {
                    http;
                    hostServer = y.c.hostServer;
                    host = this.hostServer + "/rest/memories";
                    constructor(e) {
                        this.http = e
                    }
                    addMemory(e, o) {
                        const i = new FormData;
                        return i.append("image", o, e), i.append("caption", e), this.http.post(this.host, i).pipe((0, v.T)(r => r.data), (0, g.W)(r => {
                            throw r
                        }))
                    }
                    get() {
                        return this.http.get(this.host + "/").pipe((0, v.T)(e => e.data), (0, g.W)(e => {
                            throw e
                        }))
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(s.KVO(E.Qq))
                    };
                    static \u0275prov = s.jDH({
                        token: n,
                        factory: n.\u0275fac,
                        providedIn: "root"
                    })
                }
                return n
            })();

            function Xc(n, a) {
                if (1 & n && (t.j41(0, "a", 8)(1, "button", 9), t.nrm(2, "i", 10), t.k0s()()), 2 & n) {
                    const e = t.XpG().$implicit,
                        o = t.XpG(2);
                    t.Y8G("href", t.ai1("https://twitter.com/intent/tweet?text=", e.caption, " ", o.twitterHandle, "&hashtags=appsec"), t.B4B)
                }
            }

            function Wc(n, a) {
                if (1 & n && (t.j41(0, "span", 5), t.nrm(1, "img", 6), t.j41(2, "div", 7)(3, "div"), t.EFF(4), t.k0s(), t.nVh(5, Xc, 3, 3, "a", 8), t.k0s()()), 2 & n) {
                    const e = a.$implicit,
                        o = t.XpG(2);
                    t.R7$(), t.Y8G("src", t.mNQ(e.url), t.B4B)("alt", t.mNQ(e.caption)), t.R7$(3), t.JRh(e.caption), t.R7$(), t.vxM(o.twitterHandle ? 5 : -1)
                }
            }

            function zc(n, a) {
                if (1 & n && (t.j41(0, "div", 3), t.Z7z(1, Wc, 6, 6, "span", 5, t.fX1), t.k0s()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(), t.Dyx(e.slideshowDataSource)
                }
            }

            function Jc(n, a) {
                1 & n && (t.j41(0, "mat-card", 4)(1, "div", 2), t.nrm(2, "img", 11), t.j41(3, "mat-card-title")(4, "span", 12), t.EFF(5, " NO_SEARCH_RESULT "), t.k0s()(), t.j41(6, "mat-card-content")(7, "span", 12), t.EFF(8, " EMPTY_MEMORY_LIST "), t.k0s()()()())
            }

            function Qc(n, a) {
                if (1 & n && (t.j41(0, "div", 18), t.nrm(1, "img", 22), t.k0s()), 2 & n) {
                    const e = t.XpG(2);
                    t.R7$(), t.Y8G("src", e.imagePreview, t.B4B)("alt", e.form.value.caption)
                }
            }

            function qc(n, a) {
                1 & n && (t.j41(0, "mat-error", 14), t.EFF(1), t.nI1(2, "translate"), t.k0s()), 2 & n && (t.R7$(), t.JRh(t.bMT(2, 1, "MANDATORY_CAPTION")))
            }

            function Kc(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "div")(1, "div", 13)(2, "h2", 14), t.EFF(3, "LABEL_SHARE_A_MEMORY"), t.k0s(), t.j41(4, "form", 15)(5, "div")(6, "button", 16), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.sdS(10);
                        return s.Njj(i.click())
                    }), t.EFF(7), t.nI1(8, "translate"), t.k0s(), t.j41(9, "input", 17, 0), t.bIt("change", function(i) {
                        s.eBV(e);
                        const r = t.XpG();
                        return s.Njj(r.onImagePicked(i))
                    }), t.k0s()(), t.nVh(11, Qc, 2, 2, "div", 18), t.j41(12, "mat-form-field", 19)(13, "mat-label", 14), t.EFF(14, "LABEL_CAPTION"), t.k0s(), t.nrm(15, "input", 20), t.nVh(16, qc, 3, 3, "mat-error", 14), t.k0s(), t.j41(17, "button", 21), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.save())
                    }), t.j41(18, "mat-icon"), t.EFF(19, "send"), t.k0s(), t.EFF(20), t.nI1(21, "translate"), t.k0s()()()()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.R7$(4), t.Y8G("formGroup", e.form), t.R7$(3), t.SpI("", t.bMT(8, 6, "LABEL_PICK_IMAGE"), " "), t.R7$(4), t.vxM("" !== e.imagePreview && e.imagePreview && e.form.get("image").valid ? 11 : -1), t.R7$(5), t.vxM(e.form.get("caption").invalid ? 16 : -1), t.R7$(), t.Y8G("disabled", e.form.get("image").invalid || e.form.get("caption").invalid), t.R7$(3), t.SpI(" ", t.bMT(21, 8, "BTN_SUBMIT"), " ")
                }
            }
            S.Yv.add(N.HQ1);
            let Zc = (() => {
                class n {
                    photoWallService;
                    configurationService;
                    snackBarHelperService;
                    emptyState = !0;
                    imagePreview;
                    form = new l.J3({
                        image: new l.hs("", {
                            validators: [l.k0.required],
                            asyncValidators: [Hc]
                        }),
                        caption: new l.hs("", [l.k0.required])
                    });
                    slideshowDataSource = [];
                    twitterHandle = null;
                    constructor(e, o, i) {
                        this.photoWallService = e, this.configurationService = o, this.snackBarHelperService = i
                    }
                    ngOnInit() {
                        this.slideshowDataSource = [], this.photoWallService.get().pipe((0, g.W)(e => (console.log(e), yt.w))).subscribe(e => {
                            this.emptyState = 0 === e.length;
                            for (const o of e) o.User?.username && (o.caption = `${o.caption} (\xa9 ${o.User.username})`), this.slideshowDataSource.push({
                                url: o.imagePath,
                                caption: o.caption
                            })
                        }), this.configurationService.getApplicationConfiguration().pipe((0, g.W)(e => (console.log(e), yt.w))).subscribe(e => {
                            e?.application?.social && e.application.social.twitterUrl && (this.twitterHandle = e.application.social.twitterUrl.replace("https://twitter.com/", "@"))
                        })
                    }
                    onImagePicked(e) {
                        const o = e.target.files[0];
                        this.form.patchValue({
                            image: o
                        }), this.form.get("image").updateValueAndValidity();
                        const i = new FileReader;
                        i.onload = () => {
                            this.imagePreview = i.result
                        }, i.readAsDataURL(o)
                    }
                    save() {
                        this.photoWallService.addMemory(this.form.value.caption, this.form.value.image).subscribe({
                            next: () => {
                                this.resetForm(), this.ngOnInit(), this.snackBarHelperService.open("IMAGE_UPLOAD_SUCCESS", "confirmBar")
                            },
                            error: e => {
                                this.snackBarHelperService.open(e.error?.error, "errorBar"), console.log(e)
                            }
                        })
                    }
                    isLoggedIn() {
                        return localStorage.getItem("token")
                    }
                    resetForm() {
                        this.form.get("image").setValue(""), this.form.get("image").markAsPristine(), this.form.get("image").markAsUntouched(), this.form.get("caption").setValue(""), this.form.get("caption").markAsPristine(), this.form.get("caption").markAsUntouched(), this.form.get("caption").setErrors(null)
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(Oe), t.rXU(M.s), t.rXU(L.r))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-photo-wall"]
                        ],
                        decls: 9,
                        vars: 5,
                        consts: [
                            ["filePicker", ""],
                            ["appearance", "outlined", 1, "heading", "mat-elevation-z6", "mat-own-card", 2, "margin-bottom", "10px"],
                            [1, "mdc-card"],
                            [1, "grid"],
                            ["appearance", "outlined", 1, "mat-elevation-z0", "emptyState"],
                            [1, "container", "mat-elevation-z6"],
                            [1, "image", 3, "src", "alt"],
                            [1, "overlay"],
                            ["target", "_blank", 3, "href"],
                            ["mat-icon-button", "", "aria-label", "Tweet"],
                            [1, "fab", "fa-twitter", "fa-lg"],
                            ["alt", " No results found", "src", "assets/public/images/products/no-results.png", 1, "img-responsive", "noResult"],
                            ["translate", "", 1, "noResultText"],
                            [2, "margin-top", "10px"],
                            ["translate", ""],
                            ["enctype", "multipart/form-data", 3, "formGroup"],
                            ["mat-stroked-button", "", "type", "button", 3, "click"],
                            ["type", "file", "name", "file", 3, "change"],
                            [1, "image-preview"],
                            ["appearance", "outline", "color", "accent"],
                            ["formControlName", "caption", "type", "text", "matInput", ""],
                            ["id", "submitButton", "mat-raised-button", "", "color", "primary", 3, "click", "disabled"],
                            [3, "src", "alt"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "mat-card", 1)(1, "div", 2)(2, "h1"), t.EFF(3), t.nI1(4, "translate"), t.k0s(), t.j41(5, "div"), t.nVh(6, zc, 3, 0, "div", 3)(7, Jc, 9, 0, "mat-card", 4), t.k0s(), t.nVh(8, Kc, 22, 10, "div"), t.k0s()()), 2 & o && (t.R7$(3), t.JRh(t.bMT(4, 3, "LABEL_PHOTO_WALL")), t.R7$(3), t.vxM(i.emptyState ? 7 : 6), t.R7$(2), t.vxM(i.isLoggedIn() ? 8 : -1))
                        },
                        dependencies: [h.Hu, h.RN, h.m2, h.dh, V.iY, d.h, d.Mm, l.YN, l.qT, l.me, l.BC, l.cb, l.X1, l.j4, l.JD, b.Hl, b.$z, I.R, _.rl, _.nJ, _.TL, T.fS, T.fg, d.D9],
                        styles: ["\n\n\n\nmat-form-field[_ngcontent-%COMP%]{width:100%}.mat-own-card[_ngcontent-%COMP%]{display:block;margin-left:10%;margin-right:10%}#submitButton[_ngcontent-%COMP%]{margin-left:20%;margin-top:10px;width:60%}input[type=file][_ngcontent-%COMP%]{visibility:hidden}.image-preview[_ngcontent-%COMP%]{height:5rem;margin:1rem 0}.image-preview[_ngcontent-%COMP%]   img[_ngcontent-%COMP%]{height:100%}.heading-main[_ngcontent-%COMP%]{font-size:x-large;justify-content:center}.main-wrapper[_ngcontent-%COMP%]{margin-left:10%;margin-right:10%}.grid[_ngcontent-%COMP%]{align-items:center;display:grid;grid-gap:20px;grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}.grid[_ngcontent-%COMP%]   img[_ngcontent-%COMP%]{border-radius:4px;box-shadow:2px 2px 6px #0000004d;max-width:100%}.container[_ngcontent-%COMP%]{position:relative}.image[_ngcontent-%COMP%]{display:block}.overlay[_ngcontent-%COMP%]{bottom:0;font-size:20px;left:0;opacity:0;padding:20px;position:absolute;right:0;text-align:center;transition:.5s ease}.container[_ngcontent-%COMP%]:hover   .overlay[_ngcontent-%COMP%]{opacity:1}.emptyState[_ngcontent-%COMP%]{display:block;height:auto!important;margin:20px auto;width:50%}.noResult[_ngcontent-%COMP%]{display:block;margin-left:auto;margin-right:auto;width:60%}.noResultText[_ngcontent-%COMP%]{display:block;margin-top:10px;text-align:center}.tweet-logo[_ngcontent-%COMP%]{height:24px}"]
                    })
                }
                return n
            })();
            const td = n => ({
                    appname: n
                }),
                ed = (n, a) => a.name;

            function nd(n, a) {
                if (1 & n && (t.j41(0, "div", 0)(1, "span", 14), t.EFF(2), t.k0s()()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(2), t.JRh(e.error)
                }
            }

            function od(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "span"), t.EFF(1), t.k0s(), t.j41(2, "button", 15), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.upgradeToDeluxe())
                    }), t.j41(3, "span", 16), t.EFF(4, "LABEL_BECOME_MEMBER"), t.k0s()()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.R7$(), t.SpI(" ", e.membershipCost, "\xa4 ")
                }
            }

            function id(n, a) {
                if (1 & n && (t.j41(0, "mat-card", 13), t.nrm(1, "mat-icon", 17), t.j41(2, "div", 9)(3, "strong", 18), t.EFF(4), t.nI1(5, "translate"), t.k0s(), t.j41(6, "span"), t.EFF(7), t.nI1(8, "translate"), t.k0s()()()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(), t.Y8G("fontIcon", e.icon), t.R7$(3), t.JRh(t.bMT(5, 3, e.name)), t.R7$(3), t.JRh(t.bMT(8, 5, e.description))
                }
            }
            let ad = (() => {
                class n {
                    router;
                    userService;
                    configurationService;
                    route;
                    ngZone;
                    io;
                    membershipCost = 0;
                    error = void 0;
                    applicationName = "OWASP Juice Shop";
                    logoSrc = "assets/public/images/JuiceShop_Logo.png";
                    SHOWCASES = [{
                        icon: "slideshow",
                        name: "LABEL_DEALS_OFFERS",
                        description: "DESCRIPTION_DEALS_OFFERS"
                    }, {
                        icon: "directions_car",
                        name: "LABEL_FREE_FAST_DELIVERY",
                        description: "DESCRIPTION_FREE_FAST_DELIVERY"
                    }, {
                        icon: "add",
                        name: "LABEL_UNLIMITED_PURCHASE",
                        description: "DESCRIPTION_UNLIMITED_PURCHASE"
                    }];
                    constructor(e, o, i, r, c, p) {
                        this.router = e, this.userService = o, this.configurationService = i, this.route = r, this.ngZone = c, this.io = p
                    }
                    ngOnInit() {
                        this.configurationService.getApplicationConfiguration().subscribe({
                            next: e => {
                                const o = this.route.snapshot.queryParams.testDecal;
                                if (e?.application && (e.application.name && (this.applicationName = e.application.name), e.application.logo)) {
                                    let i = e.application.logo;
                                    "http" === i.substring(0, 4) && (i = decodeURIComponent(i.substring(i.lastIndexOf("/") + 1))), this.logoSrc = `assets/public/images/${o||i}`
                                }
                                o && this.ngZone.runOutsideAngular(() => {
                                    this.io.socket().emit("verifySvgInjectionChallenge", o)
                                })
                            },
                            error: e => {
                                console.log(e)
                            }
                        }), this.userService.deluxeStatus().subscribe({
                            next: e => {
                                this.membershipCost = e.membershipCost
                            },
                            error: e => {
                                this.error = e.error.error
                            }
                        })
                    }
                    upgradeToDeluxe() {
                        var e = this;
                        this.ngZone.run((0, F.A)(function*() {
                            return yield e.router.navigate(["/payment", "deluxe"])
                        }))
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(R.Ix), t.rXU(P), t.rXU(M.s), t.rXU(R.nX), t.rXU(t.SKi), t.rXU(et))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-deluxe-user"]
                        ],
                        decls: 18,
                        vars: 10,
                        consts: [
                            [1, "heading", "mat-elevation-z6"],
                            ["appearance", "outlined", 1, "mat-elevation-z6", "deluxe-membership"],
                            ["preserveAspectRatio", "xMidYMid meet", "viewBox", "0 0 720 720", "xmlns", "http://www.w3.org/2000/svg"],
                            ["href", "assets/public/images/deluxe/blankBoxes.png", "x", "0", "y", "0", "height", "720", "width", "720"],
                            ["x", "260", "y", "130", "height", "50"],
                            ["x", "230", "y", "330", "height", "70"],
                            ["x", "70", "y", "355", "height", "40"],
                            ["x", "120", "y", "450", "height", "55"],
                            ["x", "500", "y", "410", "height", "45"],
                            [1, "card-text"],
                            ["translate", "", 1, "item-name"],
                            ["translate", "", 3, "translateParams"],
                            [1, "feature-cards-container"],
                            ["appearance", "outlined", 1, "mat-elevation-z6", "feature-card"],
                            [1, "error"],
                            ["aria-label", "Become deluxe member", "color", "primary", "mat-button", "", "mat-raised-button", "", 1, "btn-become-member", 3, "click"],
                            ["translate", ""],
                            [3, "fontIcon"],
                            [1, "item-name"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.nVh(0, nd, 3, 1, "div", 0), t.j41(1, "mat-card", 1), s.qSk(), t.j41(2, "svg", 2), t.nrm(3, "image", 3)(4, "image", 4)(5, "image", 5)(6, "image", 6)(7, "image", 7)(8, "image", 8), t.k0s(), s.joV(), t.j41(9, "div", 9)(10, "strong", 10), t.EFF(11, "LABEL_DELUXE_MEMBERSHIP"), t.k0s(), t.j41(12, "span", 11), t.EFF(13, "DESCRIPTION_DELUXE_MEMBERSHIP"), t.k0s(), t.nVh(14, od, 5, 1), t.k0s()(), t.j41(15, "div", 12), t.Z7z(16, id, 9, 7, "mat-card", 13, ed), t.k0s()), 2 & o && (t.vxM(i.error ? 0 : -1), t.R7$(4), t.BMQ("href", i.logoSrc), t.R7$(), t.BMQ("href", i.logoSrc), t.R7$(), t.BMQ("href", i.logoSrc), t.R7$(), t.BMQ("href", i.logoSrc), t.R7$(), t.BMQ("href", i.logoSrc), t.R7$(4), t.Y8G("translateParams", t.eq3(8, td, i.applicationName)), t.R7$(2), t.vxM(i.error ? -1 : 14), t.R7$(2), t.Dyx(i.SHOWCASES))
                        },
                        dependencies: [h.Hu, h.RN, d.h, d.Mm, b.Hl, b.$z, f.m_, f.An, d.D9],
                        styles: ["\n\n\n\n[_nghost-%COMP%]{display:flex;flex-direction:column;gap:10px;margin-left:10%;margin-right:10%}.heading[_ngcontent-%COMP%]{background:#0003;font-size:x-large;padding:16px 24px}.deluxe-membership[_ngcontent-%COMP%]{align-items:center;display:grid;gap:16px;grid-template-columns:minmax(min-content,600px) auto}.deluxe-membership[_ngcontent-%COMP%]   strong[_ngcontent-%COMP%]{font-size:1.5em}.deluxe-membership[_ngcontent-%COMP%]   .card-text[_ngcontent-%COMP%]{align-items:center;display:flex;flex-direction:column;gap:10px;padding:16px;text-align:center}.deluxe-membership[_ngcontent-%COMP%]   .btn-become-member[_ngcontent-%COMP%]{width:min-content}@media screen and (max-width: 768px){.deluxe-membership[_ngcontent-%COMP%]{grid-template-columns:1fr}}.feature-cards-container[_ngcontent-%COMP%]{display:grid;gap:10px;grid-template-columns:repeat(3,1fr)}@media screen and (max-width: 768px){.feature-cards-container[_ngcontent-%COMP%]{grid-template-columns:1fr}}.feature-card[_ngcontent-%COMP%]{align-items:center;display:grid;gap:10px;grid-template-columns:max-content auto;padding:16px}.feature-card[_ngcontent-%COMP%]   .card-text[_ngcontent-%COMP%]{display:flex;flex-direction:column;gap:6px;padding:0 16px;text-align:center}.feature-card[_ngcontent-%COMP%]   mat-icon[_ngcontent-%COMP%]{font-size:72px;height:72px;width:72px}"]
                    })
                }
                return n
            })();
            var Pe = m(3246);

            function rd(n, a) {
                if (1 & n && (t.j41(0, "div")(1, "div")(2, "mat-card-title"), t.EFF(3), t.nI1(4, "translate"), t.k0s()(), t.j41(5, "div", 5), t.EFF(6), t.nI1(7, "translate"), t.j41(8, "a", 6), t.EFF(9, "8343D2"), t.k0s()(), t.nrm(10, "mat-divider", 7), t.j41(11, "div", 8)(12, "div", 9), t.EFF(13), t.nI1(14, "translate"), t.k0s(), t.nrm(15, "mat-divider", 7), t.j41(16, "p", 10), t.EFF(17, "0x8343d2eb2B13A2495De435a1b15e85b98115Ce05"), t.k0s()(), t.j41(18, "div", 8)(19, "div", 9), t.EFF(20), t.nI1(21, "translate"), t.k0s(), t.nrm(22, "mat-divider", 7)(23, "p", 11), t.nI1(24, "translate"), t.k0s()()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(3), t.JRh(t.bMT(4, 5, "TITLE_SBT")), t.R7$(3), t.SpI(" ", t.bMT(7, 7, "OWNED_BY"), " "), t.R7$(7), t.JRh(t.bMT(14, 9, "LABEL_ACCOUNT_ADDRESS")), t.R7$(7), t.JRh(t.bMT(21, 11, "LABEL_DESCRIPTION")), t.R7$(3), t.Y8G("innerHTML", t.i5U(24, 13, "NFT_SBT_BOX_TEXT", e.i18nParams), t.npT)
                }
            }

            function sd(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "div")(1, "div")(2, "mat-card-title", 12), t.EFF(3), t.nI1(4, "translate"), t.k0s()(), t.j41(5, "form", 13), t.bIt("ngSubmit", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.submitForm())
                    }), t.j41(6, "div", 14)(7, "mat-form-field", 15)(8, "mat-label"), t.EFF(9), t.nI1(10, "translate"), t.k0s(), t.j41(11, "input", 16), t.nI1(12, "translate"), t.mxI("ngModelChange", function(i) {
                        s.eBV(e);
                        const r = t.XpG();
                        return t.DH7(r.privateKey, i) || (r.privateKey = i), s.Njj(i)
                    }), t.k0s()(), t.j41(13, "h5", 17), t.EFF(14), t.k0s()(), t.j41(15, "button", 18), t.EFF(16), t.nI1(17, "translate"), t.k0s()()()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.R7$(3), t.JRh(t.bMT(4, 6, "TITLE_SBT")), t.R7$(6), t.JRh(t.bMT(10, 8, "LABEL_PRIVATE_KEY")), t.R7$(2), t.R50("ngModel", e.privateKey), t.BMQ("aria-label", t.bMT(12, 10, "LABEL_PRIVATE_KEY")), t.R7$(3), t.JRh(e.errorMessage), t.R7$(2), t.SpI(" ", t.bMT(17, 12, "BTN_AUTHENTICATE"), " ")
                }
            }
            let ld = (() => {
                class n {
                    keysService;
                    privateKey;
                    formSubmitted = !1;
                    successResponse = !1;
                    errorMessage = "";
                    i18nParams = {
                        link: '<a target="_blank" rel="noopener noreferrer" href="https://testnets.opensea.io/assets/mumbai/0xf4817631372dca68a25a18eb7a0b36d54f3dbcf7/0">Opensea</a>'
                    };
                    constructor(e) {
                        this.keysService = e
                    }
                    ngOnInit() {
                        this.checkChallengeStatus()
                    }
                    checkChallengeStatus() {
                        this.keysService.nftUnlocked().subscribe({
                            next: e => {
                                this.successResponse = e.status
                            },
                            error: e => {
                                console.error(e), this.successResponse = !1
                            }
                        })
                    }
                    submitForm() {
                        this.formSubmitted = !0, this.keysService.submitKey(this.privateKey).subscribe({
                            next: e => {
                                e.success ? (this.successResponse = !0, this.errorMessage = e.message) : this.successResponse = !1
                            },
                            error: e => {
                                this.successResponse = !1, this.errorMessage = e.error.message
                            }
                        })
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(Pe.o))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-nft-unlock"]
                        ],
                        decls: 10,
                        vars: 4,
                        consts: [
                            ["appearance", "outlined"],
                            [1, "mdc-card", "container"],
                            ["mat-raised-button", "", "color", "warn", 1, "detail-container", "offer-container", "warning-container", "align-self-center"],
                            [1, "detail-container", "offer-container", "align-self-center"],
                            ["src", "assets/public/images/products/juicy_chatbot.jpg"],
                            [1, "owner-text"],
                            ["target", "_blank", "href", "https://testnets.opensea.io/0x8343d2eb2B13A2495De435a1b15e85b98115Ce05"],
                            [1, "detail-divider"],
                            [1, "detail-box", "mat-elevation-z6"],
                            [1, "box-title"],
                            [1, "box-text"],
                            [1, "box-text", 3, "innerHTML"],
                            [1, "sbt-title"],
                            [3, "ngSubmit"],
                            [1, "form-group"],
                            ["color", "accent", "appearance", "outline", 1, "responsive-field"],
                            ["matInput", "", "id", "privateKey", "name", "privateKey", "required", "", "type", "text", 1, "text-ellipsis", 3, "ngModelChange", "ngModel"],
                            [1, "error"],
                            ["type", "submit", "mat-raised-button", "", "color", "accent", 1, "btn", "btn-primary"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "mat-card", 0)(1, "div", 1)(2, "button", 2), t.EFF(3), t.nI1(4, "translate"), t.k0s(), t.j41(5, "div", 3)(6, "div"), t.nrm(7, "img", 4), t.k0s(), t.nVh(8, rd, 25, 16, "div")(9, sd, 18, 14, "div"), t.k0s()()()), 2 & o && (t.R7$(3), t.SpI(" ", t.bMT(4, 2, "BTN_SBT_NOTE"), " "), t.R7$(5), t.vxM(i.successResponse ? 8 : 9))
                        },
                        dependencies: [h.Hu, h.RN, h.dh, b.Hl, b.$z, d.h, l.YN, l.qT, l.me, l.BC, l.cb, l.YS, l.vS, l.cV, I.R, _.rl, _.nJ, T.fS, T.fg, U.q, d.D9],
                        styles: [".container[_ngcontent-%COMP%]{display:flex;flex-direction:column;gap:20px;justify-content:center}.heading[_ngcontent-%COMP%]{justify-content:center;margin-bottom:10px;padding:12px 20px}.detail-container.offer-container[_ngcontent-%COMP%]{align-items:center;display:flex;flex-direction:row;gap:16px;max-width:700px;min-width:300px;width:70%}@media (max-width: 1279.98px){.detail-container.offer-container[_ngcontent-%COMP%]{flex-direction:column}}.align-self-center[_ngcontent-%COMP%]{align-self:center}.owner-text[_ngcontent-%COMP%]{margin-bottom:10px}.responsive-field[_ngcontent-%COMP%]{width:100%}@media (min-width: 600px){.responsive-field[_ngcontent-%COMP%]{max-width:350px;width:75vw}}.responsive-field[_ngcontent-%COMP%]   .text-ellipsis[_ngcontent-%COMP%]{overflow:hidden;text-overflow:ellipsis}.warning-container[_ngcontent-%COMP%]{align-items:center;border-radius:4px;display:flex;font-weight:600;justify-content:center;min-height:48px;padding:16px;text-align:center;white-space:normal}@media (max-width: 600px){.warning-container[_ngcontent-%COMP%]{font-size:.9em;line-height:1.4;min-height:56px;padding:20px 16px}}.sbt-title[_ngcontent-%COMP%]{font-size:1.5rem;margin:0 0 16px}@media screen and (max-width: 600px){.sbt-title[_ngcontent-%COMP%]{font-size:1.25rem}}mat-card[_ngcontent-%COMP%]{margin-bottom:10px}mat-card[_ngcontent-%COMP%]:nth-child(1){border-radius:5px;margin:0 auto;width:80%}.detail-box[_ngcontent-%COMP%]{margin-bottom:16px;padding:16px}.box-title[_ngcontent-%COMP%]{font-size:16px;font-weight:500}.box-text[_ngcontent-%COMP%]{margin-top:8px;overflow-wrap:break-word}.form-group[_ngcontent-%COMP%]{display:flex;flex-direction:column;margin-top:16px;width:100%}"]
                    })
                }
                return n
            })();
            var cd = m(4572);
            const It = Object.freeze({
                categories: [],
                difficulties: [],
                tags: [],
                status: null,
                searchQuery: null,
                showDisabledChallenges: !0,
                restrictToTutorialChallengesFirst: !1
            });

            function je(n) {
                return {
                    searchQuery: n.searchQuery ?? void 0,
                    difficulties: n.difficulties.join(",") || void 0,
                    status: n.status ?? void 0,
                    tags: n.tags.join(",") || void 0,
                    categories: n.categories.join(",") || void 0,
                    showDisabledChallenges: n.showDisabledChallenges ? void 0 : "false"
                }
            }
            let md = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/snippets";
                        constructor(e) {
                            this.http = e
                        }
                        get(e) {
                            return this.http.get(`${this.host}/${e}`).pipe((0, v.T)(o => o), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })(),
                pd = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/snippets/fixes";
                        constructor(e) {
                            this.http = e
                        }
                        get(e) {
                            return this.http.get(this.host + `/${e}`).pipe((0, v.T)(o => o), (0, g.W)(o => {
                                throw o
                            }))
                        }
                        check(e, o) {
                            return this.http.post(this.host, {
                                key: e,
                                selectedFix: o
                            }).pipe((0, v.T)(i => i), (0, g.W)(i => {
                                throw i
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })(),
                ud = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/snippets/verdict";
                        constructor(e) {
                            this.http = e
                        }
                        check(e, o) {
                            return this.http.post(this.host, {
                                key: e,
                                selectedLines: o
                            }).pipe((0, v.T)(i => i), (0, g.W)(i => {
                                throw i
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })();
            const hd = ["codeComponent"],
                gd = (n, a) => a.index;

            function fd(n, a) {
                if (1 & n && t.nrm(0, "td-ngx-text-diff", 1, 0), 2 & n) {
                    const e = t.XpG().$implicit,
                        o = t.XpG();
                    t.Y8G("left", o.snippet)("right", null == e ? null : e.fix)("format", o.format)
                }
            }

            function _d(n, a) {
                if (1 & n && t.nVh(0, fd, 2, 3, "td-ngx-text-diff", 1), 2 & n) {
                    const e = a.$index,
                        o = t.XpG();
                    t.vxM(o.selectedFix === e ? 0 : -1)
                }
            }
            let Cd = (() => {
                class n {
                    cookieService;
                    differs;
                    differ;
                    constructor(e, o) {
                        this.cookieService = e, this.differs = o, this.cookieService = e, this.differ = this.differs.find({}).create()
                    }
                    snippet = "";
                    fixes = [];
                    selectedFix = 0;
                    randomFixes = [];
                    format = "SideBySide";
                    codeComponent;
                    ngOnInit() {
                        this.cookieService.hasKey("code-fixes-component-format") ? this.format = this.cookieService.get("code-fixes-component-format") : (this.format = "LineByLine", this.cookieService.put("code-fixes-component-format", "LineByLine"))
                    }
                    ngDoCheck() {
                        try {
                            const e = this.differ.diff({
                                "diff-format": this.codeComponent.format
                            });
                            e && e.forEachChangedItem(o => {
                                this.format = o.currentValue, this.cookieService.put("code-fixes-component-format", this.format)
                            })
                        } catch {}
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU($.O1), t.rXU(D.MKu))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-code-fixes"]
                        ],
                        viewQuery: function(o, i) {
                            if (1 & o && t.GBs(hd, 5), 2 & o) {
                                let r;
                                t.mGM(r = t.lsd()) && (i.codeComponent = r.first)
                            }
                        },
                        inputs: {
                            snippet: "snippet",
                            fixes: "fixes",
                            selectedFix: "selectedFix",
                            randomFixes: "randomFixes",
                            format: "format"
                        },
                        decls: 2,
                        vars: 0,
                        consts: [
                            ["codeComponent", ""],
                            [3, "left", "right", "format"]
                        ],
                        template: function(o, i) {
                            1 & o && t.Z7z(0, _d, 1, 1, null, null, gd), 2 & o && t.Dyx(i.randomFixes)
                        },
                        dependencies: [Ut.eX, Ut.F_],
                        styles: ["[_nghost-%COMP%]{display:block;padding-top:16px}"]
                    })
                }
                return n
            })();
            var Ot = m(6369);

            function bd(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "div", 5), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG();
                        return s.Njj(r.selectLines(i.lineNumber))
                    }), t.EFF(1), t.k0s()
                }
                if (2 & n) {
                    const e = a.$implicit;
                    t.Y8G("id", "line" + e.lineNumber), t.R7$(), t.JRh(e.marked ? "\u2705" : "\u{1f532}")
                }
            }
            let vd = (() => {
                class n {
                    _code = "";
                    get code() {
                        return this._code
                    }
                    set code(e) {
                        this._code = e || ""
                    }
                    vulnLines;
                    lineMarkers;
                    addLine = new t.bkB;
                    langs = ["javascript", "typescript", "json", "yaml"];
                    ngOnInit() {
                        this.lineMarkers = this.code.split("\n").map((e, o) => ({
                            lineNumber: o + 1,
                            marked: !1
                        }))
                    }
                    selectLines(e) {
                        const o = this.lineMarkers[e - 1];
                        o.marked = !o.marked;
                        const i = [];
                        for (const {
                                marked: r,
                                lineNumber: c
                            }
                            of this.lineMarkers) r && i.push(c);
                        this.addLine.emit(i)
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-code-area"]
                        ],
                        inputs: {
                            code: "code",
                            vulnLines: "vulnLines"
                        },
                        outputs: {
                            addLine: "addLine"
                        },
                        decls: 6,
                        vars: 3,
                        consts: [
                            ["id", "code-area"],
                            ["id", "code"],
                            ["id", "emphasize"],
                            [1, "lineMarker", 3, "id"],
                            [3, "highlight", "lineNumbers", "languages"],
                            [1, "lineMarker", 3, "click", "id"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "div", 0)(1, "pre", 1)(2, "div", 2), t.Z7z(3, bd, 2, 2, "div", 3, t.fX1), t.k0s(), t.nrm(5, "code", 4), t.k0s()()), 2 & o && (t.R7$(3), t.Dyx(i.lineMarkers), t.R7$(2), t.Y8G("highlight", i.code)("lineNumbers", !0)("languages", i.langs))
                        },
                        dependencies: [Ot.fw, Ot.f4],
                        styles: ["pre[_ngcontent-%COMP%]   code.hljs[_ngcontent-%COMP%]{display:block;overflow-x:auto;padding:1em}code.hljs[_ngcontent-%COMP%]{padding:3px 5px}.hljs[_ngcontent-%COMP%]{background:#fff;color:#000}.hljs-comment[_ngcontent-%COMP%], .hljs-quote[_ngcontent-%COMP%], .hljs-variable[_ngcontent-%COMP%]{color:green}.hljs-keyword[_ngcontent-%COMP%], .hljs-selector-tag[_ngcontent-%COMP%], .hljs-built_in[_ngcontent-%COMP%], .hljs-name[_ngcontent-%COMP%], .hljs-tag[_ngcontent-%COMP%]{color:#00f}.hljs-string[_ngcontent-%COMP%], .hljs-title[_ngcontent-%COMP%], .hljs-section[_ngcontent-%COMP%], .hljs-attribute[_ngcontent-%COMP%], .hljs-literal[_ngcontent-%COMP%], .hljs-template-tag[_ngcontent-%COMP%], .hljs-template-variable[_ngcontent-%COMP%], .hljs-type[_ngcontent-%COMP%], .hljs-addition[_ngcontent-%COMP%]{color:#a31515}.hljs-deletion[_ngcontent-%COMP%], .hljs-selector-attr[_ngcontent-%COMP%], .hljs-selector-pseudo[_ngcontent-%COMP%], .hljs-meta[_ngcontent-%COMP%]{color:#2b91af}.hljs-doctag[_ngcontent-%COMP%]{color:gray}.hljs-attr[_ngcontent-%COMP%]{color:red}.hljs-symbol[_ngcontent-%COMP%], .hljs-bullet[_ngcontent-%COMP%], .hljs-link[_ngcontent-%COMP%]{color:#00b0e8}.hljs-emphasis[_ngcontent-%COMP%]{font-style:italic}.hljs-strong[_ngcontent-%COMP%]{font-weight:700}#emphasize[_ngcontent-%COMP%]{float:left;padding:1em .5em}.lineMarker[_ngcontent-%COMP%]{color:#fff;cursor:pointer}"]
                    })
                }
                return n
            })();

            function kd(n, a) {
                if (1 & n && (t.j41(0, "span", 15), t.EFF(1), t.nI1(2, "translate"), t.k0s(), t.j41(3, "mat-icon", 16), t.EFF(4), t.k0s()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(), t.JRh(t.bMT(2, 3, "TAB_FIND_IT")), t.R7$(2), t.Y8G("color", e.lockColor()), t.R7$(), t.JRh(e.lockIcon())
                }
            }

            function Sd(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "app-code-area", 17), t.bIt("addLine", function(i) {
                        s.eBV(e);
                        const r = t.XpG();
                        return s.Njj(r.addLine(i))
                    }), t.k0s()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.Y8G("code", null == e.snippet ? null : e.snippet.snippet)("vulnLines", null == e.snippet ? null : e.snippet.vulnLines)
                }
            }

            function Fd(n, a) {
                1 & n && (t.j41(0, "pre")(1, "code", 18), t.EFF(2, "LOADING_CODE_SNIPPET"), t.k0s()())
            }

            function xd(n, a) {
                if (1 & n && (t.j41(0, "span", 15), t.EFF(1), t.nI1(2, "translate"), t.k0s(), t.j41(3, "mat-icon", 16), t.EFF(4), t.k0s()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(), t.JRh(t.bMT(2, 3, "TAB_FIX_IT")), t.R7$(2), t.Y8G("color", e.lockColor()), t.R7$(), t.JRh(e.lockIcon())
                }
            }

            function Td(n, a) {
                if (1 & n && t.nrm(0, "app-code-fixes", 5), 2 & n) {
                    const e = t.XpG();
                    t.Y8G("snippet", null == e.snippet ? null : e.snippet.snippet)("fixes", e.fixes)("randomFixes", e.randomFixes)("selectedFix", e.selectedFix)
                }
            }

            function yd(n, a) {
                if (1 & n && (t.j41(0, "mat-card", 6)(1, "div", 19), t.EFF(2), t.k0s()()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(2), t.JRh(e.hint)
                }
            }

            function Rd(n, a) {
                if (1 & n && (t.j41(0, "mat-card", 20)(1, "div", 19), t.EFF(2), t.k0s()()), 2 & n) {
                    const e = t.XpG();
                    t.HbH("warn" === e.resultColor() ? "warn-notification" : "accent-notification"), t.R7$(2), t.JRh(e.explanation)
                }
            }

            function wd(n, a) {
                1 & n && t.nrm(0, "div", 9)
            }

            function Ed(n, a) {
                if (1 & n && (t.j41(0, "option", 23), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$index,
                        o = t.XpG(3);
                    t.Y8G("value", e)("selected", o.selectedFix === e), t.R7$(), t.SpI("Fix ", e + 1, " ")
                }
            }

            function Md(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "mat-form-field", 21)(1, "mat-label", 18), t.EFF(2, "LABEL_CORRECT_FIX"), t.k0s(), t.j41(3, "select", 22), t.bIt("change", function(i) {
                        s.eBV(e);
                        const r = t.XpG(2);
                        return s.Njj(r.changeFix(i))
                    }), t.Z7z(4, Ed, 2, 3, "option", 23, t.fX1), t.k0s()()
                }
                if (2 & n) {
                    const e = t.XpG(2);
                    t.R7$(4), t.Dyx(e.randomFixes)
                }
            }

            function Id(n, a) {
                if (1 & n && t.nVh(0, Md, 6, 0, "mat-form-field", 21), 2 & n) {
                    const e = t.XpG();
                    t.vxM(1 === e.tab.value ? 0 : -1)
                }
            }

            function Od(n, a) {
                if (1 & n && (t.j41(0, "a", 11)(1, "button", 24)(2, "mat-icon", 25), t.EFF(3, "thumb_up"), t.k0s()()()), 2 & n) {
                    const e = t.XpG();
                    t.Y8G("href", "https://docs.google.com/forms/d/e/1FAIpQLSdaNEuz0dzFA2sexCa0AJ4QOb2OYdEL04eQOLFD2Y4T-BW6ag/viewform?usp=pp_url&entry.384948954=" + e.dialogData.name + "&entry.435235279=Coding+Challenge&entry.1734944650=Yes", t.B4B)
                }
            }

            function Pd(n, a) {
                if (1 & n && (t.j41(0, "a", 11)(1, "button", 24)(2, "mat-icon", 26), t.EFF(3, "thumb_down"), t.k0s()()()), 2 & n) {
                    const e = t.XpG();
                    t.Y8G("href", "https://docs.google.com/forms/d/e/1FAIpQLSdaNEuz0dzFA2sexCa0AJ4QOb2OYdEL04eQOLFD2Y4T-BW6ag/viewform?usp=pp_url&entry.384948954=" + e.dialogData.name + "&entry.435235279=Coding+Challenge&entry.1734944650=No", t.B4B)
                }
            }

            function jd(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 27), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.checkLines())
                    }), t.j41(1, "span"), t.EFF(2), t.nI1(3, "translate"), t.k0s(), t.j41(4, "mat-icon", 28), t.EFF(5), t.k0s()()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.Y8G("disabled", e.solved.findIt), t.R7$(2), t.JRh(t.bMT(3, 4, "BTN_SUBMIT")), t.R7$(2), t.Y8G("color", e.resultColor()), t.R7$(), t.SpI(" ", e.resultIcon(), " ")
                }
            }

            function Ad(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 29), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.checkFix())
                    }), t.j41(1, "span"), t.EFF(2), t.nI1(3, "translate"), t.k0s(), t.j41(4, "mat-icon", 28), t.EFF(5), t.k0s()()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.Y8G("disabled", e.solved.fixIt), t.R7$(2), t.JRh(t.bMT(3, 4, "BTN_SUBMIT")), t.R7$(2), t.Y8G("color", e.resultColor()), t.R7$(), t.SpI(" ", e.resultIcon(), " ")
                }
            }
            var A = function(n) {
                return n[n.Undecided = 0] = "Undecided", n[n.Right = 1] = "Right", n[n.Wrong = 2] = "Wrong", n
            }(A || {});
            let Dd = (() => {
                    class n {
                        dialogData;
                        configurationService;
                        codeSnippetService;
                        vulnLinesService;
                        codeFixesService;
                        challengeService;
                        cookieService;
                        snippet = null;
                        fixes = null;
                        selectedLines;
                        selectedFix = 0;
                        tab = new l.hs(0);
                        lock = A.Undecided;
                        result = A.Undecided;
                        hint = null;
                        explanation = null;
                        solved = {
                            findIt: !1,
                            fixIt: !1
                        };
                        showFeedbackButtons = !0;
                        randomFixes = [];
                        constructor(e, o, i, r, c, p, k) {
                            this.dialogData = e, this.configurationService = o, this.codeSnippetService = i, this.vulnLinesService = r, this.codeFixesService = c, this.challengeService = p, this.cookieService = k
                        }
                        ngOnInit() {
                            this.configurationService.getApplicationConfiguration().subscribe({
                                next: e => {
                                    this.showFeedbackButtons = e.challenges.showFeedbackButtons
                                },
                                error: e => {
                                    console.log(e)
                                }
                            }), this.codeSnippetService.get(this.dialogData.key).subscribe({
                                next: e => {
                                    this.snippet = e, this.solved.findIt = !1, this.dialogData.codingChallengeStatus >= 1 && (this.result = A.Right, this.lock = A.Right, this.solved.findIt = !0)
                                },
                                error: e => {
                                    this.snippet = {
                                        snippet: e.error
                                    }
                                }
                            }), this.codeFixesService.get(this.dialogData.key).subscribe({
                                next: e => {
                                    this.fixes = e.fixes, this.fixes && this.shuffle(), this.solved.fixIt = this.dialogData.codingChallengeStatus >= 2
                                },
                                error: () => {
                                    this.fixes = null
                                }
                            })
                        }
                        addLine = e => {
                            this.selectedLines = e
                        };
                        setFix = e => {
                            this.selectedFix = e, this.explanation = null
                        };
                        changeFix(e) {
                            this.setFix(parseInt(e.target.value, 10))
                        }
                        toggleTab = e => {
                            this.tab.setValue(e), this.result = A.Undecided, 0 === e && this.solved.findIt && (this.result = A.Right), 1 === e && this.solved.fixIt && (this.result = A.Right)
                        };
                        checkFix = () => {
                            this.codeFixesService.check(this.dialogData.key, this.randomFixes[this.selectedFix].index).subscribe(e => {
                                this.setVerdict(e.verdict), this.explanation = e.explanation
                            })
                        };
                        checkLines = () => {
                            this.vulnLinesService.check(this.dialogData.key, this.selectedLines).subscribe(e => {
                                this.setVerdict(e.verdict), this.hint = e.hint
                            })
                        };
                        lockIcon() {
                            if (null === this.fixes) return "lock";
                            switch (this.lock) {
                                case A.Right:
                                    return "lock_open";
                                case A.Wrong:
                                case A.Undecided:
                                    return "lock"
                            }
                        }
                        lockColor() {
                            switch (this.lockIcon()) {
                                case "lock_open":
                                    return "accent";
                                case "lock":
                                    return "warn"
                            }
                        }
                        shuffle() {
                            this.randomFixes = this.fixes.map((e, o) => ({
                                fix: e,
                                index: o,
                                sort: Math.random()
                            })).sort((e, o) => e.sort - o.sort).map(({
                                fix: e,
                                index: o
                            }) => ({
                                fix: e,
                                index: o
                            }))
                        }
                        setVerdict = e => {
                            this.result !== A.Right && (e ? (0 === this.tab.value ? (this.solved.findIt = !0, this.challengeService.continueCodeFindIt().subscribe({
                                next: o => {
                                    if (!o) throw new Error("Received invalid continue code from the server!");
                                    const i = new Date;
                                    i.setFullYear(i.getFullYear() + 1), this.cookieService.put("continueCodeFindIt", o, {
                                        expires: i
                                    })
                                },
                                error: o => {
                                    console.log(o)
                                }
                            })) : (this.solved.fixIt = !0, this.challengeService.continueCodeFixIt().subscribe({
                                next: o => {
                                    if (!o) throw new Error("Received invalid continue code from the server!");
                                    const i = new Date;
                                    i.setFullYear(i.getFullYear() + 1), this.cookieService.put("continueCodeFixIt", o, {
                                        expires: i
                                    })
                                },
                                error: o => {
                                    console.log(o)
                                }
                            })), this.result = A.Right, this.lock = A.Right, m.e(705).then(m.bind(m, 4705)).then(o => {
                                o.shootConfetti()
                            }).then(() => {
                                0 === this.tab.value && null !== this.fixes && this.toggleTab(1)
                            })) : this.result = A.Wrong)
                        };
                        resultIcon() {
                            switch (this.result) {
                                case A.Right:
                                    return "check";
                                case A.Wrong:
                                    return "clear";
                                default:
                                    return "send"
                            }
                        }
                        resultColor() {
                            switch (this.resultIcon()) {
                                case "check":
                                    return "accent";
                                case "clear":
                                    return "warn"
                            }
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(w.Vh), t.rXU(M.s), t.rXU(md), t.rXU(ud), t.rXU(pd), t.rXU(Z), t.rXU($.O1))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["code-snippet"]
                            ],
                            hostAttrs: [1, "code-snippet"],
                            decls: 29,
                            vars: 19,
                            consts: [
                                ["mat-dialog-title", ""],
                                ["mat-stretch-tabs", "", 3, "selectedIndexChange", "selectedIndex"],
                                ["mat-tab-label", ""],
                                [3, "code", "vulnLines"],
                                [3, "disabled"],
                                [3, "snippet", "fixes", "randomFixes", "selectedFix"],
                                ["appearance", "outlined", 1, "primary-notification"],
                                ["appearance", "outlined", 3, "class"],
                                [1, "button-row"],
                                [2, "height", "78.25px"],
                                [1, "button-group"],
                                ["target", "_blank", 3, "href"],
                                ["mat-stroked-button", "", "mat-dialog-close", "", "aria-label", "Close Dialog", "id", "findItCloseButton", 1, "close-dialog", "buttons", 3, "mat-dialog-close"],
                                ["mat-stroked-button", "", "id", "findItSubmitButton", 3, "disabled"],
                                ["mat-stroked-button", "", "id", "fixItSubmitButton", 3, "disabled"],
                                [1, "tab-header-text"],
                                [1, "materaial-icons-outlined", 3, "color"],
                                [3, "addLine", "code", "vulnLines"],
                                ["translate", ""],
                                [1, "mdc-card"],
                                ["appearance", "outlined"],
                                ["appearance", "fill"],
                                ["matNativeControl", "", 3, "change"],
                                [3, "value", "selected"],
                                ["mat-icon-button", ""],
                                ["color", "accent"],
                                ["color", "warn"],
                                ["mat-stroked-button", "", "id", "findItSubmitButton", 3, "click", "disabled"],
                                [3, "color"],
                                ["mat-stroked-button", "", "id", "fixItSubmitButton", 3, "click", "disabled"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "h2", 0), t.EFF(1), t.nI1(2, "translate"), t.k0s(), t.j41(3, "mat-dialog-content")(4, "mat-tab-group", 1), t.bIt("selectedIndexChange", function(c) {
                                    return i.toggleTab(c)
                                }), t.j41(5, "mat-tab"), t.DNE(6, kd, 5, 5, "ng-template", 2), t.nVh(7, Sd, 1, 2, "app-code-area", 3)(8, Fd, 3, 0, "pre"), t.k0s(), t.j41(9, "mat-tab", 4), t.DNE(10, xd, 5, 5, "ng-template", 2), t.nVh(11, Td, 1, 4, "app-code-fixes", 5), t.k0s()()(), t.j41(12, "mat-dialog-actions"), t.nVh(13, yd, 3, 1, "mat-card", 6), t.nVh(14, Rd, 3, 3, "mat-card", 7), t.j41(15, "div", 8), t.nVh(16, wd, 1, 0, "div", 9)(17, Id, 1, 1), t.j41(18, "div", 10), t.nVh(19, Od, 4, 1, "a", 11), t.nVh(20, Pd, 4, 1, "a", 11), t.j41(21, "button", 12)(22, "mat-icon"), t.EFF(23, "close"), t.k0s(), t.j41(24, "span"), t.EFF(25), t.nI1(26, "translate"), t.k0s()(), t.nVh(27, jd, 6, 6, "button", 13), t.nVh(28, Ad, 6, 6, "button", 14), t.k0s()()()), 2 & o && (t.R7$(), t.Lme("", t.bMT(2, 15, "TITLE_CODING_CHALLENGE"), ": ", i.dialogData.name), t.R7$(3), t.Y8G("selectedIndex", i.tab.value), t.R7$(3), t.vxM(null !== i.snippet ? 7 : 8), t.R7$(2), t.Y8G("disabled", "lock" === i.lockIcon()), t.R7$(2), t.vxM(null !== i.snippet && null !== i.fixes ? 11 : -1), t.R7$(2), t.vxM(i.hint && 0 === i.tab.value ? 13 : -1), t.R7$(), t.vxM(i.explanation && 1 === i.tab.value ? 14 : -1), t.R7$(2), t.vxM(0 === i.tab.value ? 16 : 17), t.R7$(3), t.vxM(i.showFeedbackButtons && i.solved.fixIt ? 19 : -1), t.R7$(), t.vxM(i.showFeedbackButtons && i.solved.fixIt ? 20 : -1), t.R7$(), t.Y8G("mat-dialog-close", i.solved), t.R7$(4), t.JRh(t.bMT(26, 17, "BTN_CLOSE")), t.R7$(2), t.vxM(0 === i.tab.value ? 27 : -1), t.R7$(), t.vxM(1 === i.tab.value ? 28 : -1))
                            },
                            dependencies: [w.BI, w.Yi, bt.T8, bt.mq, vd, d.h, d.Mm, bt.ES, f.m_, f.An, Cd, w.E7, h.Hu, h.RN, I.R, _.rl, _.nJ, T.fS, T.fg, l.YN, l.xH, l.y7, V.iY, b.Hl, b.$z, w.tx, d.D9],
                            styles: ["\n\n\n\n[_nghost-%COMP%]{display:block;width:65em}.tab-header-text[_ngcontent-%COMP%]{color:var(--theme-text)}mat-dialog-actions[_ngcontent-%COMP%]{--mat-dialog-actions-padding: 20px 24px;display:grid;gap:16px;grid-template-columns:1fr}.button-row[_ngcontent-%COMP%]{align-items:center;display:flex;justify-content:space-between}.button-group[_ngcontent-%COMP%]{align-items:center;display:flex;gap:8px;justify-content:flex-end}"]
                        })
                    }
                    return n
                })(),
                $d = (() => {
                    class n {
                        http;
                        hostServer = y.c.hostServer;
                        host = this.hostServer + "/api/Hints";
                        constructor(e) {
                            this.http = e
                        }
                        getAll() {
                            return this.http.get(this.host + "/").pipe((0, v.T)(e => e.data), (0, g.W)(e => {
                                throw e
                            }))
                        }
                        put(e, o) {
                            return this.http.put(`${this.host}/${e}`, o).pipe((0, v.T)(i => i.data), (0, g.W)(i => {
                                throw i
                            }))
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO(E.Qq))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })();
            var Ae = m(869);
            const Ud = () => [];

            function Vd(n, a) {
                1 & n && (s.qSk(), t.rj2(0, "svg", 0), t.Hgh(1, "path", 1), t.eux())
            }
            let De = (() => {
                class n {
                    difficulty;
                    static \u0275fac = function(o) {
                        return new(o || n)
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["difficulty-stars"]
                        ],
                        inputs: {
                            difficulty: "difficulty"
                        },
                        decls: 2,
                        vars: 1,
                        consts: [
                            ["width", "18", "height", "18", "viewBox", "0 0 14 13", "fill", "none", "xmlns", "http://www.w3.org/2000/svg"],
                            ["d", "M6.56386 0.511963L8.10332 5.24992H13.0851L9.05475 8.17813L10.5942 12.9161L6.56386 9.98787L2.53352 12.9161L4.07297 8.17813L0.0426283 5.24992H5.02441L6.56386 0.511963Z", "fill", "var(--theme-text)"]
                        ],
                        template: function(o, i) {
                            1 & o && t.Z7z(0, Vd, 2, 0, ":svg:svg", 0, t.fX1), 2 & o && t.Dyx(t.lJ4(0, Ud).constructor(i.difficulty))
                        },
                        styles: ["[_nghost-%COMP%]{display:flex}"]
                    })
                }
                return n
            })();
            const $e = n => ({
                    env: n
                }),
                Gd = (n, a) => ({
                    "partially-completed": n,
                    completed: a
                }),
                Yd = n => ({
                    completed: n
                }),
                Hd = (n, a) => ({
                    next: n,
                    total: a
                });

            function Xd(n, a) {
                if (1 & n && (t.j41(0, "span", 8), t.nI1(1, "translate"), t.EFF(2), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.Y8G("matTooltip", t.bMT(1, 2, "TAG_" + (null == e ? null : e.toUpperCase().split(" ").join("_")) + "_DESCRIPTION")), t.R7$(2), t.JRh(e)
                }
            }

            function Wd(n, a) {
                if (1 & n && (t.j41(0, "button", 10), t.nI1(1, "translate"), t.j41(2, "mat-icon"), t.EFF(3, "info_outline"), t.k0s()()), 2 & n) {
                    const e = t.XpG();
                    t.Y8G("matTooltip", t.i5U(1, 3, "CHALLENGE_UNAVAILABLE", t.eq3(6, $e, e.challenge.disabledEnv))), t.R7$(2), t.xc7("color", "var(--theme-warn)")
                }
            }

            function zd(n, a) {
                if (1 & n && (t.j41(0, "button", 10), t.nI1(1, "translate"), t.j41(2, "mat-icon"), t.EFF(3, "info_outline"), t.k0s()()), 2 & n) {
                    const e = t.XpG();
                    t.Y8G("matTooltip", t.i5U(1, 3, "CHALLENGE_UNAVAILABLE_SAFETYMODE", t.eq3(6, $e, e.challenge.disabledEnv))), t.R7$(2), t.xc7("color", "var(--theme-warn)")
                }
            }

            function Jd(n, a) {
                if (1 & n && (t.j41(0, "span", 17), t.EFF(1), t.k0s()), 2 & n) {
                    const e = t.XpG(2);
                    t.R7$(), t.SpI("", e.challenge.codingChallengeStatus, "/2")
                }
            }

            function Qd(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 16), t.nI1(1, "translate"), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.openCodingChallengeDialog(i.challenge.key))
                    }), t.nVh(2, Jd, 2, 1, "span", 17), t.j41(3, "mat-icon"), t.EFF(4, "code"), t.k0s()()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.Y8G("disabled", !1 === e.challenge.solved && "always" !== e.applicationConfiguration.challenges.codingChallengesEnabled)("ngClass", t.l_i(6, Gd, 1 === e.challenge.codingChallengeStatus, 2 === e.challenge.codingChallengeStatus))("matTooltip", t.bMT(1, 4, e.challenge.solved ? "LAUNCH_CODING_CHALLENGE" : "SOLVE_HACKING_CHALLENGE")), t.R7$(2), t.vxM(0 !== e.challenge.codingChallengeStatus ? 2 : -1)
                }
            }

            function qd(n, a) {
                if (1 & n && (t.j41(0, "a", 12), t.nI1(1, "translate"), t.j41(2, "mat-icon"), t.EFF(3, "policy_outline"), t.k0s()()), 2 & n) {
                    const e = t.XpG();
                    t.Y8G("href", e.challenge.mitigationUrl, t.B4B)("matTooltip", t.bMT(1, 2, "INFO_VULNERABILITY_MITIGATION_LINK"))
                }
            }

            function Kd(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 18), t.nI1(1, "translate"), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.repeatChallengeNotification(i.challenge.key))
                    }), t.j41(2, "mat-icon"), t.EFF(3, "flag_outline"), t.k0s()()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.Y8G("ngClass", t.eq3(4, Yd, e.challenge.solved))("matTooltip", t.bMT(1, 2, "NOTIFICATION_RESEND_INSTRUCTIONS"))
                }
            }

            function Zd(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 19), t.nI1(1, "translate"), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.startHackingInstructorFor(i.challenge.name))
                    }), t.j41(2, "mat-icon"), t.EFF(3, "school_outline"), t.k0s()()
                }
                2 & n && t.Y8G("matTooltip", t.bMT(1, 1, "INFO_HACKING_INSTRUCTOR"))
            }

            function tm(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 20), t.nI1(1, "translate"), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.unlockHint(i.challenge.nextHint))
                    }), t.j41(2, "mat-icon"), t.EFF(3), t.k0s(), t.EFF(4), t.k0s()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.xc7("padding", "0 6px 0 4px"), t.Y8G("matTooltip", e.challenge.hintText + (e.challenge.nextHint ? "\n\n" + t.i5U(1, 7, "CLICK_TO_UNLOCK_NEXT_HINT", t.l_i(10, Hd, e.challenge.hintsUnlocked + 1, e.challenge.hintsAvailable)) : ""))("disabled", !e.challenge.nextHint), t.R7$(3), t.JRh(e.challenge.nextHint ? "lightbulb_outlined" : "lightbulb"), t.R7$(), t.Lme(" ", e.challenge.hintsUnlocked, "/", e.challenge.hintsAvailable, " ")
                }
            }
            let em = (() => {
                class n {
                    challenge;
                    openCodingChallengeDialog;
                    repeatChallengeNotification;
                    unlockHint;
                    applicationConfiguration;
                    hasInstructions = () => !1;
                    startHackingInstructorFor = (0, F.A)(function*() {});
                    ngOnInit() {
                        var e = this;
                        return (0, F.A)(function*() {
                            const {
                                hasInstructions: o,
                                startHackingInstructorFor: i
                            } = yield m.e(781).then(m.bind(m, 1143));
                            e.hasInstructions = o, e.startHackingInstructorFor = i
                        })()
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["challenge-card"]
                        ],
                        inputs: {
                            challenge: "challenge",
                            openCodingChallengeDialog: "openCodingChallengeDialog",
                            repeatChallengeNotification: "repeatChallengeNotification",
                            unlockHint: "unlockHint",
                            applicationConfiguration: "applicationConfiguration"
                        },
                        decls: 20,
                        vars: 11,
                        consts: [
                            [1, "category-row"],
                            [1, "name-row"],
                            [1, "dot"],
                            [1, "name"],
                            [3, "difficulty"],
                            [1, "description-row", 3, "innerHtml"],
                            [1, "bottom-row"],
                            [1, "tags"],
                            [1, "tag", 3, "matTooltip"],
                            [1, "badge-group"],
                            [1, "badge", 3, "matTooltip"],
                            [1, "badge", 3, "disabled", "ngClass", "matTooltip"],
                            ["target", "_blank", "rel", "noopener noreferrer", "aria-label", "Vulnerability mitigation link", 1, "badge", "not-completable", 3, "href", "matTooltip"],
                            [1, "badge", 3, "ngClass", "matTooltip"],
                            [1, "badge", "not-completable", 3, "matTooltip"],
                            ["matTooltipClass", "multiline-tooltip", 1, "badge", "not-completable", 3, "padding", "matTooltip", "disabled"],
                            [1, "badge", 3, "click", "disabled", "ngClass", "matTooltip"],
                            [1, "badge-status"],
                            [1, "badge", 3, "click", "ngClass", "matTooltip"],
                            [1, "badge", "not-completable", 3, "click", "matTooltip"],
                            ["matTooltipClass", "multiline-tooltip", 1, "badge", "not-completable", 3, "click", "matTooltip", "disabled"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "span", 0), t.EFF(1), t.k0s(), t.j41(2, "div", 1), t.nrm(3, "div", 2), t.j41(4, "span", 3), t.EFF(5), t.k0s(), t.nrm(6, "difficulty-stars", 4), t.k0s(), t.nrm(7, "div", 5), t.j41(8, "div", 6)(9, "div", 7), t.Z7z(10, Xd, 3, 4, "span", 8, t.fX1), t.k0s(), t.j41(12, "div", 9), t.nVh(13, Wd, 4, 8, "button", 10), t.nVh(14, zd, 4, 8, "button", 10), t.nVh(15, Qd, 5, 9, "button", 11), t.nVh(16, qd, 4, 4, "a", 12), t.nVh(17, Kd, 4, 6, "button", 13), t.nVh(18, Zd, 4, 3, "button", 14), t.nVh(19, tm, 5, 13, "button", 15), t.k0s()()), 2 & o && (t.R7$(), t.JRh(i.challenge.category), t.R7$(4), t.JRh(i.challenge.name), t.R7$(), t.Y8G("difficulty", i.challenge.difficulty), t.R7$(), t.Y8G("innerHtml", i.challenge.description, t.npT), t.R7$(3), t.Dyx(i.challenge.tagList), t.R7$(3), t.vxM(null !== i.challenge.disabledEnv && "safetyMode" !== i.challenge.disabledEnv ? 13 : -1), t.R7$(), t.vxM("safetyMode" === i.challenge.disabledEnv ? 14 : -1), t.R7$(), t.vxM("never" !== i.applicationConfiguration.challenges.codingChallengesEnabled && i.challenge.hasCodingChallenge ? 15 : -1), t.R7$(), t.vxM(i.challenge.mitigationUrl && i.challenge.solved ? 16 : -1), t.R7$(), t.vxM(i.challenge.solved && i.applicationConfiguration.ctf.showFlagsInNotifications ? 17 : -1), t.R7$(), t.vxM(i.applicationConfiguration.hackingInstructor.isEnabled && i.hasInstructions(i.challenge.name) ? 18 : -1), t.R7$(), t.vxM(i.challenge.hintsAvailable > 0 ? 19 : -1))
                        },
                        dependencies: [De, B.oV, f.m_, f.An, Y.YU, d.h, d.D9],
                        styles: ["[_nghost-%COMP%]{background-color:var(--theme-background-darker);border-radius:2px;box-shadow:0 0 0 2px #50505040;display:grid;grid-template-rows:min-content min-content auto min-content;max-height:144px;overflow:hidden;padding:12px 16px}.solved[_nghost-%COMP%]{border-top:4px solid var(--theme-accent)}.solved[_nghost-%COMP%]   .dot[_ngcontent-%COMP%]{background-color:var(--theme-accent)}.unsolved[_nghost-%COMP%]{border-top:4px solid rgba(80,80,80,.25)}.unsolved[_nghost-%COMP%]   .dot[_ngcontent-%COMP%]{background-color:#5c5c5c}.disabled[_nghost-%COMP%]{filter:opacity(.4)}.disabled[_nghost-%COMP%]   .dot[_ngcontent-%COMP%]{background-color:var(--theme-warn-fade-40)}.category-row[_ngcontent-%COMP%]{color:var(--theme-text-fade-50);font-size:12px;font-weight:600;line-height:14px}.name-row[_ngcontent-%COMP%]{align-items:center;column-gap:6px;display:grid;grid-template-columns:12px auto min-content;margin-top:8px}.name-row[_ngcontent-%COMP%]   .dot[_ngcontent-%COMP%]{border-radius:50%;height:14px;width:14px}.name-row[_ngcontent-%COMP%]   .name[_ngcontent-%COMP%]{font-size:16px;font-weight:600;line-height:18px}.difficulty[_ngcontent-%COMP%]{display:flex}.description-row[_ngcontent-%COMP%]{color:var(--theme-text-fade-50);font-size:12px;font-weight:400;line-height:14px;margin-top:12px;overflow-y:auto}.description-row[_ngcontent-%COMP%]::-webkit-scrollbar{height:10px;width:10px}.description-row[_ngcontent-%COMP%]::-webkit-scrollbar-track{background:var(--theme-primary-light);border-radius:10px}.description-row[_ngcontent-%COMP%]::-webkit-scrollbar-thumb{background:var(--theme-primary-darker);border-radius:10px}.description-row[_ngcontent-%COMP%]::-webkit-scrollbar-corner{display:none}@supports not selector(::-webkit-scrollbar){html[_ngcontent-%COMP%]{scrollbar-width:thin}}.bottom-row[_ngcontent-%COMP%]{align-items:end;column-gap:4px;display:grid;grid-template-columns:auto min-content;margin-top:16px}.bottom-row[_ngcontent-%COMP%]   .tags[_ngcontent-%COMP%]{display:flex;flex-wrap:wrap;gap:4px}.bottom-row[_ngcontent-%COMP%]   .tag[_ngcontent-%COMP%]{background:#3d3d3d;border-radius:1.82849px;color:#5fc091;font-size:10px;font-style:normal;font-weight:400;line-height:14px;padding:6px}.badge-group[_ngcontent-%COMP%]{column-gap:4px;display:flex}.badge[_ngcontent-%COMP%]{align-items:center;background-color:var(--theme-background-lighter);border:0;border-radius:12px;color:var(--theme-text)!important;color:var(--theme-text);cursor:pointer;display:flex;height:24px;justify-content:center;min-width:24px;padding:0;position:relative}.badge[_ngcontent-%COMP%]   mat-icon[_ngcontent-%COMP%]{font-size:16px;height:16px;width:16px}.badge[_ngcontent-%COMP%]   .badge-status[_ngcontent-%COMP%]{background-color:var(--theme-background-darker);border:1px solid var(--theme-background-light);border-radius:4px;font-size:10px;padding:0 5px;position:absolute;top:-35%}.badge.completed[_ngcontent-%COMP%]{background-color:var(--theme-accent)}.badge.completed[_ngcontent-%COMP%]   .badge-status[_ngcontent-%COMP%]{color:var(--theme-accent)}.badge.partially-completed[_ngcontent-%COMP%]{background-color:var(--theme-accent-dark)}.badge.partially-completed[_ngcontent-%COMP%]   .badge-status[_ngcontent-%COMP%]{color:var(--theme-accent-dark)}.badge.not-completable[_ngcontent-%COMP%]{background-color:var(--theme-accent)}.badge.not-completable[_ngcontent-%COMP%]   .badge-status[_ngcontent-%COMP%]{color:var(--theme-accent)}.badge[_ngcontent-%COMP%]:disabled{background-color:var(--theme-background-light);cursor:not-allowed}.badge[_ngcontent-%COMP%]:disabled   .badge-status[_ngcontent-%COMP%]{color:var(--theme-accent)}"]
                    })
                }
                return n
            })();
            const nm = [
                    [
                        ["", "warning-icon", ""]
                    ],
                    [
                        ["", "warning-text", ""]
                    ],
                    [
                        ["", "warning-action", ""]
                    ]
                ],
                om = ["[warning-icon]", "[warning-text]", "[warning-action]"];
            let Le = (() => {
                class n {
                    static \u0275fac = function(o) {
                        return new(o || n)
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["warning-card"]
                        ],
                        ngContentSelectors: om,
                        decls: 7,
                        vars: 0,
                        consts: [
                            [1, "warning-container"],
                            [1, "warning-text-icon-group"],
                            [1, "warning-text"],
                            [1, "warning-action"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.NAR(nm), t.rj2(0, "div", 0)(1, "div", 1), t.SdG(2), t.rj2(3, "span", 2), t.SdG(4, 1), t.eux()(), t.rj2(5, "div", 3), t.SdG(6, 2), t.eux()())
                        },
                        styles: [".warning-container[_ngcontent-%COMP%]{align-items:center;background-color:var(--theme-background-dark);border-radius:4px;display:grid;grid-template-columns:auto min-content;margin-top:16px;padding:12px}@media (max-width: 600px){.warning-container[_ngcontent-%COMP%]{grid-template-columns:auto;row-gap:12px}}.warning-text[_ngcontent-%COMP%]{margin-left:12px;padding-right:8px}.warning-text-icon-group[_ngcontent-%COMP%]{align-items:center;display:grid;grid-template-columns:min-content auto}"]
                    })
                }
                return n
            })();
            const im = n => ({
                num: n
            });

            function am(n, a) {
                if (1 & n && (t.j41(0, "warning-card")(1, "mat-icon", 0), t.EFF(2, "school"), t.k0s(), t.j41(3, "span", 1), t.EFF(4), t.nI1(5, "translate"), t.k0s()()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(4), t.SpI(" ", t.i5U(5, 1, "INFO_FULL_CHALLENGE_MODE", t.eq3(4, im, e.allChallenges.length)), " ")
                }
            }
            let rm = (() => {
                class n {
                    allChallenges;
                    applicationConfig = null;
                    tutorialModeActive = null;
                    ngOnChanges() {
                        if (!this.applicationConfig?.challenges?.restrictToTutorialsFirst) return void(this.tutorialModeActive = !1);
                        const e = this.allChallenges.filter(o => null !== o.tutorialOrder).every(o => o.solved);
                        this.tutorialModeActive = !e
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["tutorial-mode-warning"]
                        ],
                        inputs: {
                            allChallenges: "allChallenges",
                            applicationConfig: "applicationConfig"
                        },
                        features: [t.OA$],
                        decls: 1,
                        vars: 1,
                        consts: [
                            ["warning-icon", ""],
                            ["warning-text", "", 1, "tutorial-mode-warning-text"]
                        ],
                        template: function(o, i) {
                            1 & o && t.nVh(0, am, 6, 6, "warning-card"), 2 & o && t.vxM(i.tutorialModeActive ? 0 : -1)
                        },
                        dependencies: [Le, f.m_, f.An, d.h, d.D9],
                        encapsulation: 2
                    })
                }
                return n
            })();
            const sm = n => ({
                    num: n,
                    env: "safety mode being turned on"
                }),
                lm = (n, a) => ({
                    num: n,
                    env: a
                });

            function cm(n, a) {
                if (1 & n && t.nrm(0, "i", 1), 2 & n) {
                    const e = t.XpG(2);
                    t.Y8G("ngClass", "fab fa-" + e.disabledBecauseOfEnv.toString().toLowerCase())
                }
            }

            function dm(n, a) {
                if (1 & n && t.nrm(0, "i", 1), 2 & n) {
                    const e = t.XpG(2);
                    t.Y8G("ngClass", "icon-" + e.disabledBecauseOfEnv.toString().toLowerCase())
                }
            }

            function mm(n, a) {
                if (1 & n && (t.nrm(0, "span", 2), t.nI1(1, "translate")), 2 & n) {
                    const e = t.XpG(2);
                    t.Y8G("innerHTML", t.i5U(1, 1, "INFO_DISABLED_CHALLENGES", t.eq3(4, sm, e.numberOfDisabledChallenges)), t.npT)
                }
            }

            function pm(n, a) {
                if (1 & n && (t.nrm(0, "span", 2), t.nI1(1, "translate")), 2 & n) {
                    const e = t.XpG(2);
                    t.Y8G("innerHTML", t.i5U(1, 1, "INFO_DISABLED_CHALLENGES", t.l_i(4, lm, e.numberOfDisabledChallenges, e.disabledBecauseOfEnv)), t.npT)
                }
            }

            function um(n, a) {
                1 & n && (t.EFF(0), t.nI1(1, "translate")), 2 & n && t.SpI(" ", t.bMT(1, 1, "HIDE_DISABLED_CHALLENGES"), " ")
            }

            function hm(n, a) {
                1 & n && (t.EFF(0), t.nI1(1, "translate")), 2 & n && t.SpI(" ", t.bMT(1, 1, "SHOW_DISABLED_CHALLENGES"), " ")
            }

            function gm(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "warning-card"), t.qex(1, 0), t.nVh(2, cm, 1, 1, "i", 1)(3, dm, 1, 1, "i", 1), t.bVm(), t.nVh(4, mm, 2, 6, "span", 2)(5, pm, 2, 7, "span", 2), t.j41(6, "button", 3), t.bIt("click", function() {
                        s.eBV(e);
                        const i = t.XpG();
                        return s.Njj(i.toggleShowDisabledChallenges())
                    }), t.nVh(7, um, 2, 3)(8, hm, 2, 3), t.k0s()()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.R7$(2), t.vxM("Windows" === e.disabledBecauseOfEnv ? 2 : 3), t.R7$(2), t.vxM("Safety Mode" === e.disabledBecauseOfEnv ? 4 : 5), t.R7$(3), t.vxM(!0 === e.filterSetting.showDisabledChallenges ? 7 : 8)
                }
            }
            let fm = (() => {
                class n {
                    challenges;
                    filterSetting;
                    filterSettingChange = new t.bkB;
                    numberOfDisabledChallenges = 0;
                    disabledBecauseOfEnv = null;
                    disabledOnWindows;
                    numberOfDisabledChallengesOnWindows = 0;
                    ngOnChanges() {
                        const e = this.challenges.filter(i => null !== i.disabledEnv),
                            o = e.filter(i => "Windows" === i.disabledEnv);
                        this.numberOfDisabledChallenges = e.length, this.numberOfDisabledChallenges > 0 && (this.disabledBecauseOfEnv = e[0].disabledEnv), o.length > 0 && (this.disabledOnWindows = !0, this.numberOfDisabledChallengesOnWindows = o.length)
                    }
                    toggleShowDisabledChallenges() {
                        const e = {
                            ...structuredClone(this.filterSetting),
                            showDisabledChallenges: !this.filterSetting.showDisabledChallenges
                        };
                        this.filterSetting = e, this.filterSettingChange.emit(e)
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["challenges-unavailable-warning"]
                        ],
                        inputs: {
                            challenges: "challenges",
                            filterSetting: "filterSetting"
                        },
                        outputs: {
                            filterSettingChange: "filterSettingChange"
                        },
                        features: [t.OA$],
                        decls: 1,
                        vars: 1,
                        consts: [
                            ["warning-icon", ""],
                            [1, "env-icon", 3, "ngClass"],
                            ["warning-text", "", 3, "innerHTML"],
                            ["warning-action", "", "mat-button", "", "color", "accent", 3, "click"]
                        ],
                        template: function(o, i) {
                            1 & o && t.nVh(0, gm, 9, 3, "warning-card"), 2 & o && t.vxM(i.numberOfDisabledChallenges > 0 ? 0 : -1)
                        },
                        dependencies: [Le, Y.YU, b.Hl, b.$z, d.h, d.D9],
                        styles: [".env-icon[_ngcontent-%COMP%]{font-size:18px;padding:2px}"]
                    })
                }
                return n
            })();
            var _m = m(9183),
                Be = m(2578),
                Cm = m(6648);
            let bm = (() => {
                    class n {
                        cookieService;
                        challengeService;
                        snackBarHelperService;
                        snackBar;
                        VERSION = 1;
                        constructor(e, o, i, r) {
                            this.cookieService = e, this.challengeService = o, this.snackBarHelperService = i, this.snackBar = r
                        }
                        save(e = "owasp_juice_shop") {
                            const o = {
                                version: this.VERSION
                            };
                            o.banners = {
                                welcomeBannerStatus: this.cookieService.get("welcomebanner_status") ? this.cookieService.get("welcomebanner_status") : void 0,
                                cookieConsentStatus: this.cookieService.get("cookieconsent_status") ? this.cookieService.get("cookieconsent_status") : void 0
                            }, o.language = this.cookieService.get("language") ? this.cookieService.get("language") : void 0;
                            const i = this.challengeService.continueCode(),
                                r = this.challengeService.continueCodeFindIt(),
                                c = this.challengeService.continueCodeFixIt();
                            (0, wt.p)([i, r, c]).subscribe({
                                next: ([p, k, x]) => {
                                    o.continueCode = p, o.continueCodeFindIt = k, o.continueCodeFixIt = x;
                                    const O = new Blob([JSON.stringify(o)], {
                                        type: "text/plain;charset=utf-8"
                                    });
                                    (0, Be.saveAs)(O, `${e}-${(new Date).toISOString().split("T")[0]}.json`)
                                },
                                error: () => {
                                    console.log("Failed to retrieve continue code(s) for backup from server. Using cookie values as fallback."), o.continueCode = this.cookieService.get("continueCode") ? this.cookieService.get("continueCode") : void 0, o.continueCodeFindIt = this.cookieService.get("continueCodeFindIt") ? this.cookieService.get("continueCodeFindIt") : void 0, o.continueCodeFixIt = this.cookieService.get("continueCodeFixIt") ? this.cookieService.get("continueCodeFixIt") : void 0;
                                    const p = new Blob([JSON.stringify(o)], {
                                        type: "text/plain;charset=utf-8"
                                    });
                                    (0, Be.saveAs)(p, `${e}-${(new Date).toISOString().split("T")[0]}.json`)
                                }
                            })
                        }
                        restore(e) {
                            return (0, Cm.H)(e.text().then(o => {
                                const i = JSON.parse(o);
                                i.version === this.VERSION ? (this.restoreCookie("welcomebanner_status", i.banners?.welcomeBannerStatus), this.restoreCookie("cookieconsent_status", i.banners?.cookieConsentStatus), this.restoreCookie("language", i.language), this.restoreCookie("continueCodeFindIt", i.continueCodeFindIt), this.restoreCookie("continueCodeFixIt", i.continueCodeFixIt), this.restoreCookie("continueCode", i.continueCode), this.snackBar.open("Backup has been restored from " + e.name, "Apply changes now", {
                                    duration: 1e4,
                                    panelClass: ["mat-body"]
                                }).onAction().subscribe(() => {
                                    const c = i.continueCode ? this.challengeService.restoreProgress(encodeURIComponent(i.continueCode)) : (0, Mt.of)(!0),
                                        p = i.continueCodeFindIt ? this.challengeService.restoreProgressFindIt(encodeURIComponent(i.continueCodeFindIt)) : (0, Mt.of)(!0),
                                        k = i.continueCodeFixIt ? this.challengeService.restoreProgressFixIt(encodeURIComponent(i.continueCodeFixIt)) : (0, Mt.of)(!0);
                                    (0, wt.p)([c, p, k]).subscribe({
                                        next: () => {
                                            location.reload()
                                        },
                                        error: x => {
                                            console.log(x)
                                        }
                                    })
                                })) : this.snackBarHelperService.open(`Version ${i.version} is incompatible with expected version ${this.VERSION}`, "errorBar")
                            }).catch(o => {
                                this.snackBarHelperService.open(`Backup restore operation failed: ${o.message}`, "errorBar")
                            }))
                        }
                        restoreCookie(e, o) {
                            if (o) {
                                const i = new Date;
                                i.setFullYear(i.getFullYear() + 1), this.cookieService.put(e, o, {
                                    expires: i
                                })
                            } else this.cookieService.remove(e)
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(s.KVO($.O1), s.KVO(Z), s.KVO(L.r), s.KVO(rt.UG))
                        };
                        static \u0275prov = s.jDH({
                            token: n,
                            factory: n.\u0275fac,
                            providedIn: "root"
                        })
                    }
                    return n
                })(),
                vm = (() => {
                    class n {
                        localBackupService;
                        constructor(e) {
                            this.localBackupService = e
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)(t.rXU(bm))
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["score-board-additional-settings-dialog"]
                            ],
                            decls: 35,
                            vars: 22,
                            consts: [
                                ["restoreBackupFile", ""],
                                ["mat-dialog-title", "", "translate", ""],
                                [1, "settings"],
                                [1, "setting-group"],
                                [1, "setting"],
                                ["mat-flat-button", "", "color", "primary", 3, "click"],
                                [1, "tooltip"],
                                ["type", "file", 2, "display", "none", 3, "change"],
                                ["mat-flat-button", "", "cdkFocusInitial", "", 1, "close-btn", 3, "mat-dialog-close"]
                            ],
                            template: function(o, i) {
                                if (1 & o) {
                                    const r = t.RV6();
                                    t.j41(0, "mat-dialog-content")(1, "h1", 1), t.EFF(2), t.nI1(3, "translate"), t.k0s(), t.j41(4, "div", 2)(5, "div", 3)(6, "h2"), t.EFF(7), t.nI1(8, "translate"), t.k0s(), t.j41(9, "div", 4)(10, "span"), t.EFF(11), t.nI1(12, "translate"), t.k0s(), t.j41(13, "button", 5), t.bIt("click", function() {
                                        return s.eBV(r), s.Njj(i.localBackupService.save())
                                    }), t.j41(14, "mat-icon"), t.EFF(15, "save_alt"), t.k0s(), t.j41(16, "span", 6), t.EFF(17), t.nI1(18, "translate"), t.k0s()()(), t.j41(19, "div", 4)(20, "input", 7, 0), t.bIt("change", function(p) {
                                        return s.eBV(r), s.Njj(i.localBackupService.restore(p.target.files.item(0)))
                                    }), t.k0s(), t.j41(22, "span"), t.EFF(23), t.nI1(24, "translate"), t.k0s(), t.j41(25, "button", 5), t.bIt("click", function() {
                                        s.eBV(r);
                                        const p = t.sdS(21);
                                        return s.Njj(p.click())
                                    }), t.j41(26, "mat-icon"), t.EFF(27, "settings_backup_restore"), t.k0s(), t.j41(28, "span", 6), t.EFF(29), t.nI1(30, "translate"), t.k0s()()()()(), t.j41(31, "mat-dialog-actions")(32, "button", 8), t.EFF(33), t.nI1(34, "translate"), t.k0s()()()
                                }
                                2 & o && (t.R7$(2), t.JRh(t.bMT(3, 8, "SCOREBOARD_ADDITIONAL_SETTINGS")), t.R7$(5), t.JRh(t.bMT(8, 10, "BACKUP_HEADLINE")), t.R7$(4), t.JRh(t.bMT(12, 12, "SAVE_BACKUP_TOOLTIP")), t.R7$(6), t.JRh(t.bMT(18, 14, "BTN_SAVE_BACKUP")), t.R7$(6), t.JRh(t.bMT(24, 16, "RESTORE_BACKUP_TOOLTIP")), t.R7$(6), t.JRh(t.bMT(30, 18, "BTN_RESTORE_BACKUP")), t.R7$(3), t.Y8G("mat-dialog-close", null), t.R7$(), t.SpI(" ", t.bMT(34, 20, "BTN_CLOSE"), " "))
                            },
                            dependencies: [w.Yi, w.BI, d.h, d.Mm, b.Hl, b.$z, f.m_, f.An, w.E7, w.tx, d.D9],
                            styles: [".mat-mdc-radio-button[_ngcontent-%COMP%] ~ .mat-mdc-radio-button[_ngcontent-%COMP%]{margin-left:16px}mat-dialog-actions[_ngcontent-%COMP%]{justify-content:end}h1[_ngcontent-%COMP%]{font-size:24px!important;margin-bottom:24px}.settings[_ngcontent-%COMP%]{min-height:400px;overflow-y:auto}.setting-group[_ngcontent-%COMP%]{border-bottom:3px solid var(--theme-background-lighter);margin-bottom:16px}.setting-group[_ngcontent-%COMP%]   h2[_ngcontent-%COMP%]{font-size:18px;margin-bottom:8px}.setting-group[_ngcontent-%COMP%]:last-child{border-bottom:unset;margin-bottom:0}.setting[_ngcontent-%COMP%]{align-items:center;display:grid;gap:16px;grid-template-columns:auto max-content;margin-bottom:16px}button[_ngcontent-%COMP%]   mat-icon[_ngcontent-%COMP%]{margin-right:8px}h2[_ngcontent-%COMP%]{color:var(--theme-text)}span[_ngcontent-%COMP%]{color:var(--theme-text);font-size:14px}mat-icon[_ngcontent-%COMP%]{font-size:24px!important;height:24px!important;width:24px!important}.tooltip[_ngcontent-%COMP%]{font-size:14px}.close-btn[_ngcontent-%COMP%]:hover{background-color:var(--theme-background-lighter)!important}"]
                        })
                    }
                    return n
                })(),
                km = (() => {
                    class n {
                        transform(e) {
                            if (0 === e.length) return "";
                            const i = function Sm(n) {
                                const a = [];
                                let e = null;
                                for (const o of n) null === e ? e = {
                                    start: o,
                                    end: o
                                } : o === e.end + 1 ? e.end = o : (a.push(e), e = {
                                    start: o,
                                    end: o
                                });
                                return a.push(e), a
                            }((0, Ae.A)(e));
                            return function Fm(n) {
                                return n.map(a => a.start === a.end ? a.start.toString() : `${a.start} - ${a.end}`).join(", ")
                            }(i)
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)
                        };
                        static \u0275pipe = t.EJ8({
                            name: "difficultySelectionSummary",
                            type: n,
                            pure: !0
                        })
                    }
                    return n
                })();
            const Ne = n => ({
                selected: n
            });

            function xm(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "button", 2), t.nI1(1, "translate"), t.bIt("click", function() {
                        const i = s.eBV(e).$implicit,
                            r = t.XpG();
                        return s.Njj(r.toggleCategorySelected(i))
                    }), t.EFF(2), t.k0s()
                }
                if (2 & n) {
                    const e = a.$implicit,
                        o = t.XpG();
                    t.Y8G("ngClass", t.eq3(5, Ne, o.isCategorySelected(e)))("matTooltip", t.bMT(1, 3, "CATEGORY_" + e.toUpperCase().split(" ").join("_") + "_DESCRIPTION")), t.R7$(2), t.SpI(" ", e, " ")
                }
            }
            let Tm = (() => {
                class n {
                    availableCategories = new Set;
                    allChallenges;
                    categories;
                    categoriesChange = new t.bkB;
                    ngOnInit() {
                        this.availableCategories = n.getAvailableCategories(this.allChallenges)
                    }
                    ngOnChanges() {
                        this.availableCategories = n.getAvailableCategories(this.allChallenges)
                    }
                    static getAvailableCategories(e) {
                        return new Set(e.map(o => o.category))
                    }
                    toggleCategorySelected(e) {
                        this.isCategorySelected(e) ? this.categories = this.categories.filter(o => o !== e) : this.categories.push(e), this.categoriesChange.emit(this.categories)
                    }
                    isCategorySelected(e) {
                        return this.categories.includes(e)
                    }
                    isAllCategoriesSelected() {
                        return 0 === this.categories.length
                    }
                    resetCategoryFilter() {
                        this.categories = It.categories, this.categoriesChange.emit(this.categories)
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["category-filter"]
                        ],
                        inputs: {
                            allChallenges: "allChallenges",
                            categories: "categories"
                        },
                        outputs: {
                            categoriesChange: "categoriesChange"
                        },
                        features: [t.OA$],
                        decls: 4,
                        vars: 3,
                        consts: [
                            [1, "pill", "selected", 3, "click", "ngClass"],
                            [1, "pill", 3, "ngClass", "matTooltip"],
                            [1, "pill", 3, "click", "ngClass", "matTooltip"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "button", 0), t.bIt("click", function() {
                                return i.resetCategoryFilter()
                            }), t.EFF(1, " All\n"), t.k0s(), t.Z7z(2, xm, 3, 7, "button", 1, t.fX1)), 2 & o && (t.Y8G("ngClass", t.eq3(1, Ne, i.isAllCategoriesSelected())), t.R7$(2), t.Dyx(i.availableCategories))
                        },
                        dependencies: [Y.YU, B.oV, d.h, d.D9],
                        styles: ["[_nghost-%COMP%]{display:flex;flex-direction:row;flex-wrap:wrap;gap:8px;margin:24px 0}.pill[_ngcontent-%COMP%]{background-color:var(--theme-primary-darker);border:1px solid transparent;border-radius:234234px;color:var(--theme-text);cursor:pointer;font-size:14px;font-weight:600;line-height:19px;padding:8px 16px}.pill.selected[_ngcontent-%COMP%]{background-color:var(--theme-accent-dark);border:1px solid var(--theme-accent);color:var(--theme-text-invert-15)}"]
                    })
                }
                return n
            })();

            function ym(n, a) {
                1 & n && (t.EFF(0), t.nI1(1, "translate")), 2 & n && t.SpI(" ", t.bMT(1, 1, "LABEL_DIFFICULTY"), " ")
            }

            function Rm(n, a) {
                if (1 & n && (t.j41(0, "span"), t.EFF(1), t.nI1(2, "difficultySelectionSummary"), t.k0s()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(), t.SpI(" ", t.bMT(2, 1, e.filterSetting.difficulties), " ")
                }
            }

            function wm(n, a) {
                if (1 & n && (t.j41(0, "mat-option", 10), t.EFF(1), t.k0s()), 2 & n) {
                    const e = a.$implicit;
                    t.Y8G("value", e), t.R7$(), t.JRh(e)
                }
            }
            let Em = (() => {
                class n {
                    dialog;
                    allChallenges;
                    filterSetting;
                    filterSettingChange = new t.bkB;
                    reset;
                    constructor(e) {
                        this.dialog = e
                    }
                    tags = new Set;
                    ngOnChanges() {
                        this.tags = new Set(this.allChallenges.flatMap(e => e.tagList))
                    }
                    onDifficultyFilterChange(e) {
                        const o = structuredClone(this.filterSetting);
                        o.difficulties = e, this.filterSettingChange.emit(o)
                    }
                    onStatusFilterChange(e) {
                        const o = structuredClone(this.filterSetting);
                        o.status = e, this.filterSettingChange.emit(o)
                    }
                    onTagFilterChange(e) {
                        const o = structuredClone(this.filterSetting);
                        o.tags = e, this.filterSettingChange.emit(o)
                    }
                    onCategoryFilterChange(e) {
                        const o = structuredClone(this.filterSetting);
                        o.categories = e, this.filterSettingChange.emit(o)
                    }
                    onSearchQueryFilterChange(e) {
                        const o = structuredClone(this.filterSetting);
                        o.searchQuery = e, this.filterSettingChange.emit(o)
                    }
                    canBeReset() {
                        return this.filterSetting.difficulties.length > 0 || null !== this.filterSetting.status || this.filterSetting.tags.length > 0 || this.filterSetting.categories.length > 0 || !!this.filterSetting.searchQuery || !this.filterSetting.showDisabledChallenges
                    }
                    openAdditionalSettingsDialog() {
                        this.dialog.open(vm)
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(w.bZ))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["filter-settings"]
                        ],
                        inputs: {
                            allChallenges: "allChallenges",
                            filterSetting: "filterSetting",
                            reset: "reset"
                        },
                        outputs: {
                            filterSettingChange: "filterSettingChange"
                        },
                        features: [t.OA$],
                        decls: 67,
                        vars: 50,
                        consts: [
                            [1, "filter-row"],
                            [1, "search-form-field"],
                            ["matPrefix", "", 1, "search-icon"],
                            ["type", "search", "matInput", "", 3, "input", "value"],
                            [1, "options-group"],
                            ["multiple", "", 3, "selectionChange", "value"],
                            [3, "value"],
                            [3, "difficulty"],
                            ["translate", ""],
                            ["placeholder", "All", 3, "selectionChange", "value"],
                            [1, "mat-body", 3, "value"],
                            ["value", "unsolved", 1, "mat-body"],
                            ["value", "partially-solved", 1, "mat-body"],
                            ["value", "solved", 1, "mat-body"],
                            [1, "additional-settings-wrapper"],
                            [1, "reset-filters-wrapper"],
                            ["id", "reset-filters", "mat-icon-button", "", "aria-label", "reset all filters", 3, "click", "matTooltip"],
                            ["width", "16", "height", "16", "viewBox", "0 0 16 16", "fill", "none", "xmlns", "http://www.w3.org/2000/svg"],
                            ["d", "M12.2442 2.96672L9.56672 6.37922L10.6392 7.45172C11.4117 6.46922 14.3742 2.67422 14.3742 2.67422C14.7567 2.17922 14.4042 1.46672 13.7817 1.46672H4.65422L6.15422 2.96672H12.2442Z", "fill", "currentColor"],
                            ["d", "M1.63922 0.574219L0.574219 1.63172L7.03172 8.21672V12.7167C7.03172 13.1292 7.36922 13.4667 7.78172 13.4667H9.28172C9.69422 13.4667 10.0317 13.1292 10.0317 12.7167V11.0892L14.3667 15.4242L15.4242 14.3667L1.63922 0.574219Z", "fill", "currentColor"],
                            ["for", "reset-filters", 1, "reset-filters-label"],
                            ["mat-icon-button", "", "aria-label", "Open additional settings dialog", 1, "additional-settings-button", 3, "click", "matTooltip"],
                            [3, "categoriesChange", "categories", "allChallenges"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "div", 0)(1, "div", 1)(2, "mat-form-field")(3, "mat-icon", 2), t.EFF(4, "search"), t.k0s(), t.j41(5, "mat-label"), t.EFF(6), t.nI1(7, "translate"), t.k0s(), t.j41(8, "input", 3), t.bIt("input", function(c) {
                                return i.onSearchQueryFilterChange(c.target.value)
                            }), t.k0s()()(), t.j41(9, "div", 4)(10, "mat-form-field")(11, "mat-label"), t.EFF(12), t.nI1(13, "translate"), t.k0s(), t.j41(14, "mat-select", 5), t.bIt("selectionChange", function(c) {
                                return i.onDifficultyFilterChange(c.value)
                            }), t.j41(15, "mat-select-trigger"), t.nVh(16, ym, 2, 3)(17, Rm, 3, 3, "span"), t.k0s(), t.j41(18, "mat-option", 6), t.nrm(19, "difficulty-stars", 7), t.k0s(), t.j41(20, "mat-option", 6), t.nrm(21, "difficulty-stars", 7), t.k0s(), t.j41(22, "mat-option", 6), t.nrm(23, "difficulty-stars", 7), t.k0s(), t.j41(24, "mat-option", 6), t.nrm(25, "difficulty-stars", 7), t.k0s(), t.j41(26, "mat-option", 6), t.nrm(27, "difficulty-stars", 7), t.k0s(), t.j41(28, "mat-option", 6), t.nrm(29, "difficulty-stars", 7), t.k0s()()(), t.j41(30, "mat-form-field")(31, "mat-label", 8), t.EFF(32, "LABEL_STATUS"), t.k0s(), t.j41(33, "mat-select", 9), t.bIt("selectionChange", function(c) {
                                return i.onStatusFilterChange(c.value)
                            }), t.j41(34, "mat-option", 10), t.EFF(35), t.nI1(36, "translate"), t.k0s(), t.j41(37, "mat-option", 11), t.EFF(38), t.nI1(39, "translate"), t.k0s(), t.j41(40, "mat-option", 12), t.EFF(41), t.nI1(42, "translate"), t.k0s(), t.j41(43, "mat-option", 13), t.EFF(44), t.nI1(45, "translate"), t.k0s()()(), t.j41(46, "mat-form-field")(47, "mat-label"), t.EFF(48), t.nI1(49, "translate"), t.k0s(), t.j41(50, "mat-select", 5), t.bIt("selectionChange", function(c) {
                                return i.onTagFilterChange(c.value)
                            }), t.Z7z(51, wm, 2, 2, "mat-option", 10, t.fX1), t.k0s()(), t.j41(53, "div", 14)(54, "div", 15)(55, "button", 16), t.nI1(56, "translate"), t.bIt("click", function() {
                                return i.reset()
                            }), s.qSk(), t.j41(57, "svg", 17), t.nrm(58, "path", 18)(59, "path", 19), t.k0s()(), s.joV(), t.j41(60, "label", 20), t.EFF(61), t.nI1(62, "translate"), t.k0s()(), t.j41(63, "button", 21), t.bIt("click", function() {
                                return i.openAdditionalSettingsDialog()
                            }), t.j41(64, "mat-icon"), t.EFF(65, "settings"), t.k0s()()()()(), t.j41(66, "category-filter", 22), t.bIt("categoriesChange", function(c) {
                                return i.onCategoryFilterChange(c)
                            }), t.k0s()), 2 & o && (t.R7$(6), t.JRh(t.bMT(7, 32, "NAV_SEARCH")), t.R7$(2), t.Y8G("value", i.filterSetting.searchQuery), t.R7$(4), t.JRh(t.bMT(13, 34, "LABEL_DIFFICULTY")), t.R7$(2), t.Y8G("value", i.filterSetting.difficulties), t.R7$(2), t.vxM(0 === i.filterSetting.difficulties.length ? 16 : 17), t.R7$(2), t.Y8G("value", 1), t.R7$(), t.Y8G("difficulty", 1), t.R7$(), t.Y8G("value", 2), t.R7$(), t.Y8G("difficulty", 2), t.R7$(), t.Y8G("value", 3), t.R7$(), t.Y8G("difficulty", 3), t.R7$(), t.Y8G("value", 4), t.R7$(), t.Y8G("difficulty", 4), t.R7$(), t.Y8G("value", 5), t.R7$(), t.Y8G("difficulty", 5), t.R7$(), t.Y8G("value", 6), t.R7$(), t.Y8G("difficulty", 6), t.R7$(4), t.Y8G("value", i.filterSetting.status), t.R7$(), t.Y8G("value", null), t.R7$(), t.JRh(t.bMT(36, 36, "STATUS_ALL")), t.R7$(3), t.JRh(t.bMT(39, 38, "STATUS_UNSOLVED")), t.R7$(3), t.JRh(t.bMT(42, 40, "STATUS_PARTIALLY_SOLVED")), t.R7$(3), t.JRh(t.bMT(45, 42, "STATUS_SOLVED")), t.R7$(4), t.JRh(t.bMT(49, 44, "LABEL_TAGS")), t.R7$(2), t.Y8G("value", i.filterSetting.tags), t.R7$(), t.Dyx(i.tags), t.R7$(4), t.Y8G("matTooltip", t.bMT(56, 46, "LABEL_RESET_ALL_FILTERS")), t.R7$(2), t.xc7("color", i.canBeReset() ? "var(--theme-warn)" : "var(--theme-text-dark)"), t.R7$(4), t.JRh(t.bMT(62, 48, "LABEL_RESET_ALL_FILTERS")), t.R7$(2), t.Y8G("matTooltip", "Open additional settings dialog"), t.R7$(3), t.Y8G("categories", i.filterSetting.categories)("allChallenges", i.allChallenges))
                        },
                        dependencies: [I.R, _.rl, _.nJ, _.JW, f.m_, f.An, d.h, d.Mm, T.fS, T.fg, St.VO, St.$2, ue.wT, De, V.iY, B.oV, Tm, d.D9, km],
                        styles: [".options-group[_ngcontent-%COMP%]{align-items:center;column-gap:16px;display:grid;grid-template-columns:repeat(3,1fr) min-content;row-gap:8px;width:max-content}@media (max-width: 1200px){.options-group[_ngcontent-%COMP%]{width:auto}}@media (max-width: 800px){.options-group[_ngcontent-%COMP%]{grid-template-columns:auto;width:100%}}.filter-row[_ngcontent-%COMP%]{display:grid;gap:16px;grid-template-columns:auto min-content min-content;margin-top:16px}@media (max-width: 1200px){.filter-row[_ngcontent-%COMP%]{grid-template-columns:1fr}}.search-form-field[_ngcontent-%COMP%]   mat-form-field[_ngcontent-%COMP%]{width:100%}.search-icon[_ngcontent-%COMP%]{color:var(--theme-text-fade-30)}.reset-filters-wrapper[_ngcontent-%COMP%]{align-items:center;display:flex}.reset-filters-label[_ngcontent-%COMP%]{color:var(--theme-text-fade-30)}@media (min-width: 800px){.reset-filters-label[_ngcontent-%COMP%]{display:none}}.additional-settings-wrapper[_ngcontent-%COMP%]{align-items:center;display:flex}.additional-settings-button[_ngcontent-%COMP%]{align-items:center;display:flex;justify-content:center}.additional-settings-button[_ngcontent-%COMP%]   mat-icon[_ngcontent-%COMP%]{color:var(--theme-text-fade-30);font-size:16px;height:16px;line-height:16px;width:16px}"]
                    })
                }
                return n
            })();
            const Mm = [
                    [
                        ["", "right-side", ""]
                    ]
                ],
                Im = ["[right-side]"];

            function Om(n, a) {
                if (1 & n && (t.EFF(0), t.nI1(1, "number")), 2 & n) {
                    const e = t.XpG();
                    t.SpI(" ", t.i5U(1, 1, e.score / e.total * 100, "1.0-0"), "% ")
                }
            }

            function Pm(n, a) {
                if (1 & n && (t.EFF(0), t.rj2(1, "span", 5), t.EFF(2), t.eux()), 2 & n) {
                    const e = t.XpG();
                    t.SpI(" ", e.score, "/"), t.R7$(2), t.JRh(e.total)
                }
            }

            function jm(n, a) {
                if (1 & n && (t.rj2(0, "div", 3), t.Hgh(1, "div", 6), t.eux()), 2 & n) {
                    const e = t.XpG();
                    t.R7$(), t.xc7("width", e.score / e.total * 100 + "%")
                }
            }
            let Zt = (() => {
                class n {
                    description;
                    total;
                    score;
                    showAsPercentage = !0;
                    showProgressBar = !0;
                    static \u0275fac = function(o) {
                        return new(o || n)
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["score-card"]
                        ],
                        inputs: {
                            description: "description",
                            total: "total",
                            score: "score",
                            showAsPercentage: "showAsPercentage",
                            showProgressBar: "showProgressBar"
                        },
                        ngContentSelectors: Im,
                        decls: 9,
                        vars: 3,
                        consts: [
                            [1, "score-group"],
                            [1, "score"],
                            [1, "category"],
                            [1, "progress-bar-container"],
                            [1, "right-group"],
                            [1, "fraction-total"],
                            [1, "progress-bar"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.NAR(Mm), t.rj2(0, "div", 0)(1, "p", 1), t.nVh(2, Om, 2, 4)(3, Pm, 3, 2), t.eux(), t.rj2(4, "p", 2), t.EFF(5), t.eux(), t.nVh(6, jm, 2, 2, "div", 3), t.eux(), t.rj2(7, "div", 4), t.SdG(8), t.eux()), 2 & o && (t.R7$(2), t.vxM(!0 === i.showAsPercentage ? 2 : 3), t.R7$(3), t.JRh(i.description), t.R7$(), t.vxM(!0 === i.showProgressBar ? 6 : -1))
                        },
                        dependencies: [Y.QX],
                        styles: ["[_nghost-%COMP%]{background-color:var(--theme-background-dark);border-radius:4px;display:grid;grid-template-columns:auto min-content;height:100%}.score-group[_ngcontent-%COMP%]{display:flex;flex-direction:column;justify-content:center;padding:16px 0 16px 16px}.score-group[_ngcontent-%COMP%]   .score[_ngcontent-%COMP%]{color:var(--theme-text);font-size:32px;font-style:normal;font-weight:500;line-height:42px;margin-bottom:0}.score-group[_ngcontent-%COMP%]   .fraction-total[_ngcontent-%COMP%]{color:var(--theme-text-dark);font-size:16px;line-height:26px}.score-group[_ngcontent-%COMP%]   .category[_ngcontent-%COMP%]{color:var(--theme-text);font-size:20px;font-style:normal;font-weight:500;line-height:23px}.progress-bar-container[_ngcontent-%COMP%]{background-color:var(--theme-background-lighter);border-radius:3px;max-width:200px;overflow:hidden}.progress-bar-container[_ngcontent-%COMP%]   .progress-bar[_ngcontent-%COMP%]{background-color:var(--theme-accent);height:8px}"]
                    })
                }
                return n
            })();

            function Am(n, a) {
                1 & n && (s.qSk(), t.nrm(0, "stop", 8))
            }

            function Dm(n, a) {
                if (1 & n && (s.qSk(), t.nrm(0, "stop", 7)), 2 & n) {
                    const e = t.XpG().$implicit;
                    t.BMQ("offset", e.solvedChallenges / e.availableChallenges * 100 + "%")
                }
            }

            function $m(n, a) {
                if (1 & n && (t.j41(0, "div", 2)(1, "div", 3), s.qSk(), t.j41(2, "svg", 4), t.nrm(3, "path", 5), t.j41(4, "defs")(5, "linearGradient", 6), t.nrm(6, "stop", 7), t.nVh(7, Am, 1, 0, ":svg:stop", 8)(8, Dm, 1, 1, ":svg:stop", 7), t.nrm(9, "stop", 7)(10, "stop", 9), t.k0s()()(), s.joV(), t.j41(11, "span", 10), t.EFF(12), t.k0s()(), t.j41(13, "span"), t.EFF(14), t.j41(15, "span", 11), t.EFF(16), t.k0s()()()), 2 & n) {
                    const e = a.$implicit;
                    t.R7$(3), t.BMQ("fill", "url(#solved-gradient-" + e.difficulty + ")"), t.R7$(2), t.BMQ("id", "solved-gradient-" + e.difficulty), t.R7$(2), t.vxM(0 === e.availableChallenges ? 7 : 8), t.R7$(5), t.JRh(e.difficulty), t.R7$(2), t.SpI(" ", e.solvedChallenges, "/"), t.R7$(2), t.JRh(e.availableChallenges)
                }
            }
            const Lm = Object.freeze({
                1: {
                    difficulty: 1,
                    availableChallenges: 0,
                    solvedChallenges: 0
                },
                2: {
                    difficulty: 2,
                    availableChallenges: 0,
                    solvedChallenges: 0
                },
                3: {
                    difficulty: 3,
                    availableChallenges: 0,
                    solvedChallenges: 0
                },
                4: {
                    difficulty: 4,
                    availableChallenges: 0,
                    solvedChallenges: 0
                },
                5: {
                    difficulty: 5,
                    availableChallenges: 0,
                    solvedChallenges: 0
                },
                6: {
                    difficulty: 6,
                    availableChallenges: 0,
                    solvedChallenges: 0
                }
            });
            let Bm = (() => {
                    class n {
                        allChallenges = [];
                        totalChallenges;
                        solvedChallenges;
                        difficultySummaries = [{
                            difficulty: 1,
                            availableChallenges: 0,
                            solvedChallenges: 0
                        }, {
                            difficulty: 2,
                            availableChallenges: 0,
                            solvedChallenges: 0
                        }, {
                            difficulty: 3,
                            availableChallenges: 0,
                            solvedChallenges: 0
                        }, {
                            difficulty: 4,
                            availableChallenges: 0,
                            solvedChallenges: 0
                        }, {
                            difficulty: 5,
                            availableChallenges: 0,
                            solvedChallenges: 0
                        }, {
                            difficulty: 6,
                            availableChallenges: 0,
                            solvedChallenges: 0
                        }];
                        ngOnInit() {
                            this.updatedNumberOfSolvedChallenges()
                        }
                        ngOnChanges(e) {
                            this.updatedNumberOfSolvedChallenges()
                        }
                        updatedNumberOfSolvedChallenges() {
                            const e = this.allChallenges.filter(r => r.solved).length,
                                o = this.allChallenges.filter(r => r.hasCodingChallenge),
                                i = o.map(r => r.codingChallengeStatus).reduce((r, c) => r + c, 0);
                            this.difficultySummaries = n.calculateDifficultySummaries(this.allChallenges), this.totalChallenges = this.allChallenges.length + 2 * o.length, this.solvedChallenges = e + i
                        }
                        static calculateDifficultySummaries(e) {
                            const o = structuredClone(Lm);
                            for (const i of e) o[i.difficulty].availableChallenges += i.hasCodingChallenge ? 3 : 1, i.solved && (o[i.difficulty].solvedChallenges++, o[i.difficulty].solvedChallenges += i.hasCodingChallenge ? i.codingChallengeStatus : 0);
                            return Object.values(o).sort((i, r) => i.difficulty - r.difficulty)
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["difficulty-overview-score-card"]
                            ],
                            inputs: {
                                allChallenges: "allChallenges"
                            },
                            features: [t.OA$],
                            decls: 5,
                            vars: 7,
                            consts: [
                                [3, "description", "total", "score", "showAsPercentage", "showProgressBar"],
                                ["right-side", "", 1, "difficulties-group"],
                                [1, "difficulty-container"],
                                [1, "star-container"],
                                ["width", "28", "height", "25", "viewBox", "0 0 28 25", "fill", "none", "xmlns", "http://www.w3.org/2000/svg"],
                                ["d", "M12.6496 1.25874L9.57854 7.48541L2.70757 8.48713C1.4754 8.66584 0.981596 10.1849 1.87515 11.0549L6.84614 15.8989L5.67041 22.7417C5.45878 23.9786 6.76149 24.905 7.85257 24.3266L13.9993 21.0957L20.146 24.3266C21.2371 24.9003 22.5398 23.9786 22.3282 22.7417L21.1524 15.8989L26.1234 11.0549C27.017 10.1849 26.5232 8.66584 25.291 8.48713L18.42 7.48541L15.349 1.25874C14.7988 0.148847 13.2045 0.134738 12.6496 1.25874Z"],
                                ["x1", "0", "x2", "0", "y1", "1", "y2", "0"],
                                ["stop-color", "var(--theme-accent)"],
                                ["offset", "0%", "stop-color", "var(--theme-accent)"],
                                ["stop-color", "var(--theme-primary)"],
                                [1, "star-difficulty-label"],
                                [1, "total-challenges"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "score-card", 0), t.nI1(1, "translate"), t.j41(2, "div", 1), t.Z7z(3, $m, 17, 6, "div", 2, t.fX1), t.k0s()()), 2 & o && (t.Y8G("description", t.bMT(1, 5, "TITLE_CHALLENGES_SOLVED"))("total", i.totalChallenges)("score", i.solvedChallenges)("showAsPercentage", !1)("showProgressBar", !1), t.R7$(3), t.Dyx(i.difficultySummaries))
                            },
                            dependencies: [Zt, d.h, d.D9],
                            styles: ["@media (max-width: 810px){[_nghost-%COMP%]{grid-column:span 2}}@media (max-width: 600px){[_nghost-%COMP%]{grid-column:span 1}}.difficulties-group[_ngcontent-%COMP%]{background-color:var(--theme-background-darkest);border-radius:4px;color:var(--theme-text);column-gap:12px;display:grid;grid-template-columns:1fr 1fr 1fr;height:calc(100% - 24px);margin:4px;padding:8px 12px;row-gap:2px}.difficulties-group[_ngcontent-%COMP%]   .difficulty-container[_ngcontent-%COMP%]{align-items:center;display:flex;flex-direction:column;justify-content:center}.difficulties-group[_ngcontent-%COMP%]   .total-challenges[_ngcontent-%COMP%]{color:var(--theme-text-invert-30)}.star-container[_ngcontent-%COMP%]{align-items:center;display:flex;justify-content:center}.star-container[_ngcontent-%COMP%]   .star-difficulty-label[_ngcontent-%COMP%]{font-size:11px;line-height:11px;margin-top:2px;position:absolute}"]
                        })
                    }
                    return n
                })(),
                Nm = (() => {
                    class n {
                        allChallenges = [];
                        availableCodingChallenges;
                        solvedCodingChallenges;
                        ngOnInit() {
                            this.updatedNumberOfSolvedChallenges()
                        }
                        ngOnChanges(e) {
                            this.updatedNumberOfSolvedChallenges()
                        }
                        updatedNumberOfSolvedChallenges() {
                            const e = this.allChallenges.filter(o => o.hasCodingChallenge);
                            this.solvedCodingChallenges = e.map(o => o.codingChallengeStatus).reduce((o, i) => o + i, 0), this.availableCodingChallenges = 2 * e.length
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["coding-challenge-progress-score-card"]
                            ],
                            inputs: {
                                allChallenges: "allChallenges"
                            },
                            features: [t.OA$],
                            decls: 10,
                            vars: 6,
                            consts: [
                                [3, "description", "total", "score", "showAsPercentage"],
                                ["right-side", "", 1, "icon-group"],
                                ["width", "74", "height", "85", "viewBox", "0 0 74 85", "fill", "none", "xmlns", "http://www.w3.org/2000/svg"],
                                ["clip-path", "url(#clip0_2_635)"],
                                ["d", "M69.2371 41.1778C68.9506 40.8702 68.6049 40.6235 68.2209 40.4524C67.8369 40.2813 67.4224 40.1893 67.0021 40.1819C66.5817 40.1745 66.1642 40.2518 65.7744 40.4092C65.3846 40.5667 65.0305 40.801 64.7332 41.0983C64.436 41.3956 64.2016 41.7497 64.0442 42.1395C63.8867 42.5293 63.8094 42.9468 63.8168 43.3671C63.8242 43.7875 63.9162 44.202 64.0873 44.586C64.2584 44.97 64.5051 45.3156 64.8127 45.6022L65.5265 46.3201C68.3731 49.1668 70.2806 51.0785 71.512 52.698C72.6932 54.2424 72.9603 55.1439 72.9603 55.912C72.9603 56.68 72.6932 57.5816 71.512 59.1259C70.2806 60.7413 68.3731 62.6571 65.5265 65.5038L64.8127 66.2217C64.5051 66.5083 64.2584 66.8539 64.0873 67.2379C63.9162 67.6219 63.8242 68.0365 63.8168 68.4568C63.8094 68.8771 63.8867 69.2947 64.0442 69.6845C64.2016 70.0743 64.436 70.4284 64.7332 70.7256C65.0305 71.0229 65.3846 71.2572 65.7744 71.4147C66.1642 71.5721 66.5817 71.6495 67.0021 71.642C67.4224 71.6346 67.8369 71.5426 68.2209 71.3715C68.6049 71.2004 68.9506 70.9537 69.2371 70.6462L70.1095 69.778C72.76 67.1275 74.9722 64.9153 76.4915 62.9243C78.0944 60.8206 79.2213 58.6 79.2213 55.912C79.2213 53.2239 78.0944 51.0033 76.4915 48.8997C74.9722 46.9087 72.76 44.6964 70.1053 42.046L69.2371 41.1778ZM58.6269 32.7296C59.428 32.9448 60.1111 33.4692 60.5259 34.1877C60.9406 34.9062 61.0533 35.7599 60.8391 36.5613L50.0368 76.8821C49.9302 77.2792 49.7464 77.6515 49.4959 77.9776C49.2454 78.3037 48.9332 78.5772 48.577 78.7827C47.8576 79.1975 47.0029 79.3096 46.2009 79.0943C45.3988 78.879 44.7152 78.3539 44.3003 77.6345C43.8855 76.9152 43.7733 76.0605 43.9887 75.2584L54.791 34.9418C54.8975 34.5442 55.0814 34.1715 55.3321 33.845C55.5828 33.5186 55.8955 33.2447 56.2522 33.0393C56.6089 32.8338 57.0026 32.7007 57.4108 32.6475C57.819 32.5944 58.2336 32.6223 58.631 32.7296H58.6269ZM40.0192 41.1778C39.4322 40.5915 38.6366 40.2622 37.807 40.2622C36.9774 40.2622 36.1817 40.5915 35.5948 41.1778L34.7266 42.046C32.0719 44.6964 29.8597 46.9087 28.3404 48.8997C26.7375 51.0033 25.6147 53.2239 25.6147 55.912C25.6147 58.6 26.7417 60.8206 28.3404 62.9243C29.8597 64.9153 32.0719 67.1275 34.7266 69.778L35.5948 70.6462C36.1882 71.1991 36.9731 71.5002 37.7841 71.4859C38.5951 71.4716 39.3689 71.143 39.9425 70.5694C40.516 69.9959 40.8446 69.2221 40.8589 68.4111C40.8732 67.6001 40.5722 66.8152 40.0192 66.2217L39.3054 65.5038C36.4588 62.6571 34.5554 60.7454 33.3199 59.1259C32.1387 57.5816 31.8757 56.68 31.8757 55.912C31.8757 55.1439 32.1387 54.2424 33.3199 52.698C34.5554 51.0827 36.4588 49.1668 39.3054 46.3201L40.0234 45.6022C40.6096 45.0152 40.9389 44.2196 40.9389 43.39C40.9389 42.5604 40.6096 41.7647 40.0234 41.1778H40.0192Z", "fill", "var(--theme-text)", "fill-opacity", "0.12"],
                                ["fill-rule", "evenodd", "clip-rule", "evenodd", "d", "M52.4188 11.0416C27.6378 11.0416 7.54834 31.1311 7.54834 55.912C7.54834 80.693 27.6378 100.782 52.4188 100.782C77.1998 100.782 97.2892 80.693 97.2892 55.912C97.2892 31.1311 77.1998 11.0416 52.4188 11.0416ZM13.8093 55.912C13.8093 45.6722 17.8771 35.8517 25.1178 28.6111C32.3584 21.3704 42.1789 17.3026 52.4188 17.3026C62.6586 17.3026 72.4791 21.3704 79.7197 28.6111C86.9604 35.8517 91.0282 45.6722 91.0282 55.912C91.0282 66.1519 86.9604 75.9724 79.7197 83.213C72.4791 90.4537 62.6586 94.5215 52.4188 94.5215C42.1789 94.5215 32.3584 90.4537 25.1178 83.213C17.8771 75.9724 13.8093 66.1519 13.8093 55.912Z", "fill", "var(--theme-text)", "fill-opacity", "0.12"],
                                ["id", "clip0_2_635"],
                                ["width", "106", "height", "106", "fill", "var(--theme-text)"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "score-card", 0), t.nI1(1, "translate"), t.j41(2, "div", 1), s.qSk(), t.j41(3, "svg", 2)(4, "g", 3), t.nrm(5, "path", 4)(6, "path", 5), t.k0s(), t.j41(7, "defs")(8, "clipPath", 6), t.nrm(9, "rect", 7), t.k0s()()()()()), 2 & o && t.Y8G("description", t.bMT(1, 4, "TITLE_CODING_CHALLENGES"))("total", i.availableCodingChallenges)("score", i.solvedCodingChallenges)("showAsPercentage", !0)
                            },
                            dependencies: [Zt, d.h, d.D9],
                            styles: [".icon-group[_ngcontent-%COMP%]{align-items:flex-end;display:flex;height:100%;justify-content:flex-end}"]
                        })
                    }
                    return n
                })(),
                Um = (() => {
                    class n {
                        allChallenges = [];
                        solvedChallenges;
                        ngOnInit() {
                            this.updatedNumberOfSolvedChallenges()
                        }
                        ngOnChanges(e) {
                            this.updatedNumberOfSolvedChallenges()
                        }
                        updatedNumberOfSolvedChallenges() {
                            this.solvedChallenges = this.allChallenges.filter(e => e.solved).length
                        }
                        static \u0275fac = function(o) {
                            return new(o || n)
                        };
                        static \u0275cmp = t.VBU({
                            type: n,
                            selectors: [
                                ["hacking-challenge-progress-score-card"]
                            ],
                            inputs: {
                                allChallenges: "allChallenges"
                            },
                            features: [t.OA$],
                            decls: 9,
                            vars: 6,
                            consts: [
                                [3, "description", "total", "score", "showAsPercentage"],
                                ["right-side", "", 1, "icon-group"],
                                ["width", "62", "height", "69", "viewBox", "0 0 62 69", "fill", "none", "xmlns", "http://www.w3.org/2000/svg"],
                                ["clip-path", "url(#clip0_2_621)"],
                                ["d", "M51.875 72.6249H29.125M51.875 72.6249V40.7749C51.875 40.1715 51.6353 39.5929 51.2087 39.1662C50.782 38.7396 50.2034 38.4999 49.6 38.4999H31.4C30.7966 38.4999 30.218 38.7396 29.7913 39.1662C29.3647 39.5929 29.125 40.1715 29.125 40.7749V72.6249M51.875 72.6249H72.35C72.9534 72.6249 73.532 72.3852 73.9587 71.9586C74.3853 71.5319 74.625 70.9533 74.625 70.3499V61.6291C74.625 61.0257 74.3853 60.447 73.9587 60.0204C73.532 59.5937 72.9534 59.3541 72.35 59.3541H54.15C53.5466 59.3541 52.968 59.5937 52.5413 60.0204C52.1147 60.447 51.875 61.0257 51.875 61.6291V72.6249ZM29.125 72.6249V54.0457C29.125 53.4424 28.8853 52.8637 28.4587 52.4371C28.032 52.0104 27.4534 51.7707 26.85 51.7707H8.65C8.04663 51.7707 7.46798 52.0104 7.04133 52.4371C6.61469 52.8637 6.375 53.4424 6.375 54.0457V70.3499C6.375 70.9533 6.61469 71.5319 7.04133 71.9586C7.46798 72.3852 8.04663 72.6249 8.65 72.6249H29.125ZM35.9727 12.3867L39.4194 5.08015C39.5124 4.87128 39.664 4.69384 39.8558 4.56933C40.0476 4.44481 40.2713 4.37854 40.5 4.37854C40.7287 4.37854 40.9524 4.44481 41.1442 4.56933C41.336 4.69384 41.4876 4.87128 41.5806 5.08015L45.031 12.3867L52.7357 13.5659C53.7253 13.7176 54.1197 14.9916 53.403 15.7196L47.8293 21.4071L49.145 29.4378C49.3118 30.4691 48.2805 31.2578 47.3933 30.7687L40.5 26.977L33.6068 30.7687C32.7233 31.254 31.6882 30.4691 31.855 29.4378L33.1707 21.4071L27.597 15.7196C26.8765 14.9916 27.2747 13.7176 28.2605 13.5659L35.9727 12.3867Z", "stroke", "var(--theme-text)", "stroke-opacity", "0.12", "stroke-width", "5.6875", "stroke-linecap", "round", "stroke-linejoin", "round"],
                                ["id", "clip0_2_621"],
                                ["width", "62", "height", "69", "fill", "var(--theme-text)"]
                            ],
                            template: function(o, i) {
                                1 & o && (t.j41(0, "score-card", 0), t.nI1(1, "translate"), t.j41(2, "div", 1), s.qSk(), t.j41(3, "svg", 2)(4, "g", 3), t.nrm(5, "path", 4), t.k0s(), t.j41(6, "defs")(7, "clipPath", 5), t.nrm(8, "rect", 6), t.k0s()()()()()), 2 & o && t.Y8G("description", t.bMT(1, 4, "TITLE_HACKING_CHALLENGES"))("total", i.allChallenges.length)("score", i.solvedChallenges)("showAsPercentage", !0)
                            },
                            dependencies: [Zt, d.h, d.D9],
                            styles: [".icon-group[_ngcontent-%COMP%]{align-items:flex-end;display:flex;height:100%;justify-content:flex-end}"]
                        })
                    }
                    return n
                })();
            const Vm = (n, a, e) => ({
                solved: n,
                unsolved: a,
                disabled: e
            });

            function Gm(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "filter-settings", 4), t.bIt("filterSettingChange", function(i) {
                        s.eBV(e);
                        const r = t.XpG();
                        return s.Njj(r.onFilterSettingUpdate(i))
                    }), t.k0s()
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.Y8G("filterSetting", e.filterSetting)("allChallenges", e.allChallenges)("reset", e.reset.bind(e))
                }
            }

            function Ym(n, a) {
                1 & n && (t.j41(0, "div", 3), t.nrm(1, "mat-spinner"), t.k0s())
            }

            function Hm(n, a) {
                if (1 & n && t.nrm(0, "challenge-card", 9), 2 & n) {
                    const e = a.$implicit,
                        o = t.XpG(3);
                    t.Y8G("challenge", e)("applicationConfiguration", o.applicationConfiguration)("openCodingChallengeDialog", o.openCodingChallengeDialog.bind(o))("repeatChallengeNotification", o.repeatChallengeNotification.bind(o))("unlockHint", o.unlockHint.bind(o))("ngClass", t.sMw(6, Vm, e.solved, !e.solved, null !== e.disabledEnv))
                }
            }

            function Xm(n, a) {
                if (1 & n && (t.j41(0, "div", 7), t.Z7z(1, Hm, 1, 10, "challenge-card", 9, t.BUC().getChallengeKey, !0), t.k0s()), 2 & n) {
                    const e = t.XpG(2);
                    t.R7$(), t.Dyx(e.filteredChallenges)
                }
            }

            function Wm(n, a) {
                1 & n && (t.j41(0, "div", 10)(1, "p"), t.EFF(2), t.nI1(3, "translate"), t.k0s()()), 2 & n && (t.R7$(2), t.JRh(t.bMT(3, 1, "NO_CHALLENGES_FOUND")))
            }

            function zm(n, a) {
                if (1 & n && t.nVh(0, Wm, 4, 3, "div", 10), 2 & n) {
                    const e = t.XpG(2);
                    t.vxM(e.allChallenges.length > 0 ? 0 : -1)
                }
            }

            function Jm(n, a) {
                if (1 & n) {
                    const e = t.RV6();
                    t.j41(0, "challenges-unavailable-warning", 5), t.bIt("filterSettingChange", function(i) {
                        s.eBV(e);
                        const r = t.XpG();
                        return s.Njj(r.onFilterSettingUpdate(i))
                    }), t.k0s(), t.nrm(1, "tutorial-mode-warning", 6), t.nVh(2, Xm, 3, 0, "div", 7)(3, zm, 1, 1), t.nrm(4, "img", 8)
                }
                if (2 & n) {
                    const e = t.XpG();
                    t.Y8G("challenges", e.allChallenges)("filterSetting", e.filterSetting), t.R7$(), t.Y8G("allChallenges", e.allChallenges)("applicationConfig", e.applicationConfiguration), t.R7$(), t.vxM(e.filteredChallenges.length > 0 ? 2 : 3)
                }
            }
            let Qm = (() => {
                class n {
                    challengeService;
                    hintService;
                    configurationService;
                    sanitizer;
                    ngZone;
                    io;
                    dialog;
                    router;
                    route;
                    allChallenges = [];
                    filteredChallenges = [];
                    filterSetting = structuredClone(It);
                    applicationConfiguration = null;
                    isInitialized = !1;
                    subscriptions = [];
                    constructor(e, o, i, r, c, p, k, x, O) {
                        this.challengeService = e, this.hintService = o, this.configurationService = i, this.sanitizer = r, this.ngZone = c, this.io = p, this.dialog = k, this.router = x, this.route = O
                    }
                    ngOnInit() {
                        const e = (0, cd.z)([this.challengeService.find({
                            sort: "name"
                        }), this.hintService.getAll(), this.configurationService.getApplicationConfiguration()]).subscribe(([i, r, c]) => {
                            this.applicationConfiguration = c;
                            const p = i.map(k => ({
                                ...k,
                                hintText: r.filter(x => x.ChallengeId === k.id && x.unlocked).map(x => x.order + ". " + x.text).join("\n"),
                                nextHint: r.filter(x => x.ChallengeId === k.id && !x.unlocked).sort((x, O) => x.order - O.order).map(x => x.id)[0],
                                hintsUnlocked: r.filter(x => x.ChallengeId === k.id && x.unlocked).length,
                                hintsAvailable: r.filter(x => x.ChallengeId === k.id).length,
                                tagList: k.tags ? k.tags.split(",").map(x => x.trim()) : [],
                                originalDescription: k.description,
                                description: this.sanitizer.bypassSecurityTrustHtml(k.description)
                            }));
                            this.allChallenges = p, this.filterAndUpdateChallenges(), this.isInitialized = !0
                        });
                        this.subscriptions.push(e);
                        const o = this.route.queryParams.subscribe(i => {
                            this.filterSetting = function dd(n) {
                                const a = {
                                    ...structuredClone(It)
                                };
                                return n.searchQuery && (a.searchQuery = n.searchQuery), n.difficulties && (a.difficulties = n.difficulties.split(",").map(e => parseInt(e, 10)).filter(e => !isNaN(e))), n.tags && (a.tags = n.tags.split(",")), n.status && (a.status = n.status), n.categories && (a.categories = n.categories.split(",")), void 0 !== n.showDisabledChallenges && (a.showDisabledChallenges = "true" === n.showDisabledChallenges), a
                            }(i), this.filterAndUpdateChallenges()
                        });
                        this.subscriptions.push(o), this.io.socket().on("challenge solved", this.onChallengeSolvedWebsocket.bind(this)), this.io.socket().on("code challenge solved", this.onCodeChallengeSolvedWebsocket.bind(this))
                    }
                    ngOnDestroy() {
                        this.io.socket().off("challenge solved", this.onChallengeSolvedWebsocket.bind(this)), this.io.socket().off("code challenge solved", this.onCodeChallengeSolvedWebsocket.bind(this));
                        for (const e of this.subscriptions) e.unsubscribe()
                    }
                    onFilterSettingUpdate(e) {
                        this.router.navigate([], {
                            queryParams: je(e)
                        })
                    }
                    onChallengeSolvedWebsocket(e) {
                        e && (this.allChallenges = this.allChallenges.map(o => o.key === e.key ? {
                            ...o,
                            solved: !0
                        } : {
                            ...o
                        }), this.filterAndUpdateChallenges(), this.ngZone.run(() => {}))
                    }
                    onCodeChallengeSolvedWebsocket(e) {
                        e && (this.allChallenges = this.allChallenges.map(o => o.key === e.key ? {
                            ...o,
                            codingChallengeStatus: e.codingChallengeStatus
                        } : {
                            ...o
                        }), this.filterAndUpdateChallenges(), this.ngZone.run(() => {}))
                    }
                    filterAndUpdateChallenges() {
                        this.filteredChallenges = function Nd(n) {
                            return (0, Ae.A)(n, ["difficulty", "tutorialOrder", "name"])
                        }(function Ld(n, a) {
                            return n.filter(e => 0 === a.categories.length || a.categories.includes(e.category)).filter(e => 0 === a.difficulties.length || a.difficulties.includes(e.difficulty)).filter(e => 0 === a.tags.length || e.tagList.some(o => a.tags.includes(o))).filter(e => null === a.status || a.status === function Bd(n) {
                                return n.solved ? n.hasCodingChallenge ? 2 === n.codingChallengeStatus ? "solved" : "partially-solved" : n.solved ? "solved" : "unsolved" : "unsolved"
                            }(e)).filter(e => null === e.disabledEnv || a.showDisabledChallenges).filter(e => null === a.searchQuery || e.name.toLowerCase().includes(a.searchQuery.toLowerCase()) || e.originalDescription.toLowerCase().includes(a.searchQuery.toLowerCase())).filter(e => {
                                if (!a.restrictToTutorialChallengesFirst) return !0;
                                const o = n.filter(p => null !== p.tutorialOrder),
                                    i = o.every(p => p.solved);
                                if (i) return !0;
                                if (!i && null === e.tutorialOrder) return !1;
                                const r = o.filter(p => !p.solved).map(p => p.difficulty),
                                    c = Math.min(...r);
                                return e.difficulty <= c
                            })
                        }(this.allChallenges, {
                            ...this.filterSetting,
                            restrictToTutorialChallengesFirst: this.applicationConfiguration?.challenges?.restrictToTutorialsFirst ?? !0
                        }))
                    }
                    getChallengeKey(e, o) {
                        return o.key
                    }
                    reset() {
                        this.router.navigate([], {
                            queryParams: je(It)
                        })
                    }
                    openCodingChallengeDialog(e) {
                        const o = this.allChallenges.find(i => i.key === e);
                        this.dialog.open(Dd, {
                            disableClose: !0,
                            data: {
                                key: e,
                                name: o.name,
                                codingChallengeStatus: o.codingChallengeStatus
                            }
                        })
                    }
                    repeatChallengeNotification(e) {
                        var o = this;
                        return (0, F.A)(function*() {
                            const i = o.allChallenges.find(r => r.key === e);
                            yield(0, ee._)(o.challengeService.repeatNotification(encodeURIComponent(i.name)))
                        })()
                    }
                    unlockHint(e) {
                        this.hintService.put(e, {
                            unlocked: !0
                        }).subscribe({
                            next: () => {
                                this.ngOnInit()
                            },
                            error: o => {
                                console.log(o)
                            }
                        })
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)(t.rXU(Z), t.rXU($d), t.rXU(M.s), t.rXU(it.up), t.rXU(t.SKi), t.rXU(et), t.rXU(w.bZ), t.rXU(R.Ix), t.rXU(R.nX))
                    };
                    static \u0275cmp = t.VBU({
                        type: n,
                        selectors: [
                            ["app-score-board"]
                        ],
                        decls: 7,
                        vars: 6,
                        consts: [
                            [1, "score-row"],
                            [3, "allChallenges"],
                            [3, "filterSetting", "allChallenges", "reset"],
                            [1, "loading-spinner-wrapper"],
                            [3, "filterSettingChange", "filterSetting", "allChallenges", "reset"],
                            [3, "filterSettingChange", "challenges", "filterSetting"],
                            [3, "allChallenges", "applicationConfig"],
                            [1, "challenges"],
                            ["src", "assets/public/images/padding/1px.png"],
                            [3, "challenge", "applicationConfiguration", "openCodingChallengeDialog", "repeatChallengeNotification", "unlockHint", "ngClass"],
                            [1, "empty-challenges"]
                        ],
                        template: function(o, i) {
                            1 & o && (t.j41(0, "div", 0), t.nrm(1, "hacking-challenge-progress-score-card", 1)(2, "coding-challenge-progress-score-card", 1)(3, "difficulty-overview-score-card", 1), t.k0s(), t.nVh(4, Gm, 1, 3, "filter-settings", 2), t.nVh(5, Ym, 2, 0, "div", 3), t.nVh(6, Jm, 5, 5)), 2 & o && (t.R7$(), t.Y8G("allChallenges", i.allChallenges), t.R7$(), t.Y8G("allChallenges", i.allChallenges), t.R7$(), t.Y8G("allChallenges", i.allChallenges), t.R7$(), t.vxM(!1 === (null == i.applicationConfiguration ? null : i.applicationConfiguration.challenges.restrictToTutorialsFirst) ? 4 : -1), t.R7$(), t.vxM(!1 === i.isInitialized ? 5 : -1), t.R7$(), t.vxM(!0 === i.isInitialized ? 6 : -1))
                        },
                        dependencies: [Um, Nm, Bm, Em, _m.LG, fm, rm, em, Y.YU, d.h, d.D9],
                        styles: ["[_nghost-%COMP%]{display:block;margin:48px 72px}@media (max-width: 1500px){[_nghost-%COMP%]{margin:48px 64px}}@media (max-width: 900px){[_nghost-%COMP%]{margin:48px}}@media (max-width: 600px){[_nghost-%COMP%]{margin:32px 16px}}.score-row[_ngcontent-%COMP%]{column-gap:16px;display:grid;grid-template-columns:repeat(3,1fr);min-height:120px;row-gap:8px}@media (max-width: 1500px){.score-row[_ngcontent-%COMP%]{grid-template-columns:repeat(3,1fr)}}@media (max-width: 900px){.score-row[_ngcontent-%COMP%]{grid-template-columns:repeat(2,1fr)}}@media (max-width: 600px){.score-row[_ngcontent-%COMP%]{grid-template-columns:repeat(1,1fr)}}.challenges[_ngcontent-%COMP%]{column-gap:16px;display:grid;grid-template-columns:repeat(4,1fr);margin-top:32px;row-gap:12px}@media (max-width: 1500px){.challenges[_ngcontent-%COMP%]{grid-template-columns:repeat(3,1fr)}}@media (max-width: 900px){.challenges[_ngcontent-%COMP%]{grid-template-columns:repeat(2,1fr)}}@media (max-width: 600px){.challenges[_ngcontent-%COMP%]{grid-template-columns:repeat(1,1fr)}}.empty-challenges[_ngcontent-%COMP%]{align-items:center;display:flex;flex-direction:column;justify-content:center;margin:0 32px;min-height:20vh}.loading-spinner-wrapper[_ngcontent-%COMP%]{align-items:center;display:flex;justify-content:center;padding:64px;width:100%}"]
                    })
                }
                return n
            })();
            const qm = function() {
                    var n = (0, F.A)(function*() {
                        return (yield Promise.all([m.e(989), m.e(76), m.e(388)]).then(m.bind(m, 4388))).FaucetModule
                    });
                    return function() {
                        return n.apply(this, arguments)
                    }
                }(),
                Km = function() {
                    var n = (0, F.A)(function*() {
                        return (yield Promise.all([m.e(989), m.e(76), m.e(675)]).then(m.bind(m, 1675))).WalletWeb3Module
                    });
                    return function() {
                        return n.apply(this, arguments)
                    }
                }(),
                Zm = function() {
                    var n = (0, F.A)(function*() {
                        return (yield Promise.all([m.e(989), m.e(380)]).then(m.bind(m, 7380))).Web3SandboxModule
                    });
                    return function() {
                        return n.apply(this, arguments)
                    }
                }(),
                tp = [{
                    path: "administration",
                    component: cr,
                    canActivate: [ie]
                }, {
                    path: "accounting",
                    component: $l,
                    canActivate: [ae]
                }, {
                    path: "about",
                    component: Si
                }, {
                    path: "address/select",
                    component: Rs,
                    canActivate: [X]
                }, {
                    path: "address/saved",
                    component: ws,
                    canActivate: [X]
                }, {
                    path: "address/create",
                    component: we,
                    canActivate: [X]
                }, {
                    path: "address/edit/:addressId",
                    component: we,
                    canActivate: [X]
                }, {
                    path: "delivery-method",
                    component: Gc
                }, {
                    path: "deluxe-membership",
                    component: ad,
                    canActivate: [X]
                }, {
                    path: "saved-payment-methods",
                    component: pl
                }, {
                    path: "basket",
                    component: No
                }, {
                    path: "order-completion/:id",
                    component: ic
                }, {
                    path: "contact",
                    component: di
                }, {
                    path: "photo-wall",
                    component: Zc
                }, {
                    path: "complain",
                    component: xr
                }, {
                    path: "chatbot",
                    component: Er
                }, {
                    path: "order-summary",
                    component: cc
                }, {
                    path: "order-history",
                    component: Mc
                }, {
                    path: "payment/:entity",
                    component: ml
                }, {
                    path: "wallet",
                    component: pc
                }, {
                    path: "login",
                    component: Ma
                }, {
                    path: "forgot-password",
                    component: ea
                }, {
                    path: "recycle",
                    component: ns
                }, {
                    path: "register",
                    component: Yi
                }, {
                    path: "search",
                    component: Xt
                }, {
                    path: "hacking-instructor",
                    component: Xt
                }, {
                    path: "score-board",
                    component: Qm
                }, {
                    path: "track-result",
                    component: de
                }, {
                    path: "track-result/new",
                    component: de,
                    data: {
                        type: "new"
                    }
                }, {
                    path: "2fa/enter",
                    component: is
                }, {
                    path: "privacy-security",
                    component: rs,
                    children: [{
                        path: "privacy-policy",
                        component: _s
                    }, {
                        path: "change-password",
                        component: gr
                    }, {
                        path: "two-factor-authentication",
                        component: cs
                    }, {
                        path: "data-export",
                        component: gs
                    }, {
                        path: "last-login-ip",
                        component: fs
                    }]
                }, {
                    path: "juicy-nft",
                    component: ld
                }, {
                    path: "wallet-web3",
                    loadChildren: (n = (0, F.A)(function*() {
                        return yield Km()
                    }), function() {
                        return n.apply(this, arguments)
                    })
                }, {
                    path: "web3-sandbox",
                    loadChildren: function() {
                        var n = (0, F.A)(function*() {
                            return yield Zm()
                        });
                        return function() {
                            return n.apply(this, arguments)
                        }
                    }()
                }, {
                    path: "bee-haven",
                    loadChildren: function() {
                        var n = (0, F.A)(function*() {
                            return yield qm()
                        });
                        return function() {
                            return n.apply(this, arguments)
                        }
                    }()
                }, {
                    matcher: function np(n) {
                        return 0 === n.length ? null : window.location.href.includes("#access_token=") ? {
                            consumed: n
                        } : null
                    },
                    data: {
                        params: window.location.href.substr(window.location.href.indexOf("#"))
                    },
                    component: go
                }, {
                    matcher: function op(n) {
                        return 0 === n.length ? null : n[0].toString().match(function ip(...n) {
                            const a = Array.prototype.slice.call(n),
                                e = a.shift();
                            return a.reverse().map(function(o, i) {
                                return String.fromCharCode(o - e - 45 - i)
                            }).join("")
                        }(25, 184, 174, 179, 182, 186) + "sal".toLowerCase() + function ap() {
                            const a = Array.prototype.slice.call(arguments),
                                e = a.shift();
                            return a.reverse().map(function(o, i) {
                                return String.fromCharCode(o - e - 24 - i)
                            }).join("")
                        }(13, 144, 87, 152, 139, 144, 83, 138) + "a".toLowerCase()) ? {
                            consumed: n
                        } : null
                    },
                    component: ho
                }, {
                    path: "403",
                    component: as
                }, {
                    path: "**",
                    component: Xt
                }],
                ep = nt.iI.forRoot(tp, {
                    useHash: !0
                });
            var n, Ue = m(4257);
            let rp = (() => {
                class n {
                    intercept(e, o) {
                        return localStorage.getItem("token") && (e = e.clone({
                            setHeaders: {
                                Authorization: `Bearer ${localStorage.getItem("token")}`
                            }
                        })), localStorage.getItem("email") && (e = e.clone({
                            setHeaders: {
                                "X-User-Email": String(localStorage.getItem("email"))
                            }
                        })), o.handle(e)
                    }
                    static \u0275fac = function(o) {
                        return new(o || n)
                    };
                    static \u0275prov = s.jDH({
                        token: n,
                        factory: n.\u0275fac
                    })
                }
                return n
            })();
            y.c.production && (0, D.SmG)(), (0, Ue.B8)(oo, {
                providers: [(0, s.oKB)(Ue.Bb, ep, d.h.forRoot({
                    loader: {
                        provide: d.Wr,
                        useFactory: function lt(n) {
                            return new J.s(n, "./assets/i18n/", ".json")
                        },
                        deps: [E.Qq]
                    }
                }), $.lX.forRoot(), l.X1, xt.lg, Ut.eX, pt.XK, Ft.eB, Pt.FQ, dt.s5, f.m_, l.YN, I.R, St.Ve, b.Hl, Dt.vg, mo.p, u.tP, mt.Ou, h.Hu, T.fS, Nt.g7, w.hM, U.w, kt.X6, po.WX, G.MY, Bt.PO, co.u, _t.Cn, Ct.Fg, lo.Vg, so.RH, Lt.Fe, ro.Y, ot.Wk, rt._T, vt.Ez, bt.RI, $t.mV, ao.YN, io.jL, Ot.fw), {
                    provide: E.a7,
                    useClass: rp,
                    multi: !0
                }, {
                    provide: Ot.sq,
                    useValue: {
                        coreLibraryLoader: function() {
                            var n = (0, F.A)(function*() {
                                return yield m.e(300).then(m.bind(m, 5300))
                            });
                            return function() {
                                return n.apply(this, arguments)
                            }
                        }(),
                        lineNumbersLoader: function() {
                            var n = (0, F.A)(function*() {
                                return yield m.e(806).then(m.t.bind(m, 2806, 23))
                            });
                            return function() {
                                return n.apply(this, arguments)
                            }
                        }(),
                        languages: {
                            typescript: function() {
                                var n = (0, F.A)(function*() {
                                    return yield m.e(108).then(m.bind(m, 7108))
                                });
                                return function() {
                                    return n.apply(this, arguments)
                                }
                            }(),
                            javascript: function() {
                                var n = (0, F.A)(function*() {
                                    return yield m.e(236).then(m.bind(m, 8236))
                                });
                                return function() {
                                    return n.apply(this, arguments)
                                }
                            }(),
                            yaml: function() {
                                var n = (0, F.A)(function*() {
                                    return yield m.e(928).then(m.bind(m, 6928))
                                });
                                return function() {
                                    return n.apply(this, arguments)
                                }
                            }()
                        }
                    }
                }, Rt, M.s, jt, Gt, Re, P, pe, me, Tt, be, Yt, ke, Se, Vt, Fe, Q, Z, $.O1, ie, X, qt, ae, At, ye, Pe.o, st, Ht, Qt, Kt, Et, Oe, (0, E.$R)((0, E.Sx)()), (0, uo.vF)()]
            }).catch(n => console.log(n))
        },
        3246: (z, W, m) => {
            m.d(W, {
                o: () => lt
            });
            var F = m(9437),
                D = m(6354),
                s = m(5312),
                J = m(2615),
                M = m(9330);
            let lt = (() => {
                class j {
                    http;
                    hostServer = s.c.hostServer;
                    host = this.hostServer + "/rest/web3";
                    constructor(S) {
                        this.http = S
                    }
                    nftUnlocked() {
                        return this.http.get(this.host + "/nftUnlocked").pipe((0, D.T)(S => S), (0, F.W)(S => {
                            throw S
                        }))
                    }
                    nftMintListen() {
                        return this.http.get(this.host + "/nftMintListen").pipe((0, D.T)(S => S), (0, F.W)(S => {
                            throw S
                        }))
                    }
                    checkNftMinted() {
                        return this.http.get(this.hostServer + "/api/Challenges/?key=nftMintChallenge").pipe((0, D.T)(S => S), (0, F.W)(S => {
                            throw S
                        }))
                    }
                    submitKey(S) {
                        return this.http.post(this.host + "/submitKey", {
                            privateKey: S
                        }).pipe((0, D.T)(f => f), (0, F.W)(f => {
                            throw f
                        }))
                    }
                    verifyNFTWallet(S) {
                        return this.http.post(this.host + "/walletNFTVerify", {
                            walletAddress: S
                        }).pipe((0, D.T)(f => f), (0, F.W)(f => {
                            throw f
                        }))
                    }
                    walletAddressSend(S) {
                        return this.http.post(this.host + "/walletExploitAddress", {
                            walletAddress: S
                        }).pipe((0, D.T)(f => f), (0, F.W)(f => {
                            throw f
                        }))
                    }
                    static \u0275fac = function(R) {
                        return new(R || j)(J.KVO(M.Qq))
                    };
                    static \u0275prov = J.jDH({
                        token: j,
                        factory: j.\u0275fac,
                        providedIn: "root"
                    })
                }
                return j
            })()
        },
        5312: (z, W, m) => {
            m.d(W, {
                c: () => F
            });
            const F = {
                production: !0,
                hostServer: "."
            }
        },
        7916: (z, W, m) => {
            m.d(W, {
                s: () => lt
            });
            var F = m(5312),
                D = m(9437),
                s = m(6354),
                J = m(2615),
                M = m(9330);
            let lt = (() => {
                class j {
                    http;
                    hostServer = F.c.hostServer;
                    host = this.hostServer + "/rest/admin";
                    configObservable;
                    constructor(S) {
                        this.http = S
                    }
                    getApplicationConfiguration() {
                        return this.configObservable || (this.configObservable = this.http.get(this.host + "/application-configuration").pipe((0, s.T)(S => S.config, (0, D.W)(S => {
                            throw S
                        })))), this.configObservable
                    }
                    static \u0275fac = function(R) {
                        return new(R || j)(J.KVO(M.Qq))
                    };
                    static \u0275prov = J.jDH({
                        token: j,
                        factory: j.\u0275fac,
                        providedIn: "root"
                    })
                }
                return j
            })()
        },
        8559: (z, W, m) => {
            m.d(W, {
                r: () => J
            });
            var F = m(2615),
                D = m(3955),
                s = m(5416);
            let J = (() => {
                class M {
                    translateService;
                    snackBar;
                    constructor(j, y) {
                        this.translateService = j, this.snackBar = y
                    }
                    open(j, y) {
                        this.translateService.get(j).subscribe({
                            next: S => {
                                this.snackBar.open(S, "X", {
                                    duration: 5e3,
                                    panelClass: [y, "mat-body"]
                                })
                            },
                            error: () => {
                                this.snackBar.open(j, "X", {
                                    duration: 5e3,
                                    panelClass: [y, "mat-body"]
                                })
                            }
                        })
                    }
                    static \u0275fac = function(y) {
                        return new(y || M)(F.KVO(D.c$), F.KVO(s.UG))
                    };
                    static \u0275prov = F.jDH({
                        token: M,
                        factory: M.\u0275fac,
                        providedIn: "root"
                    })
                }
                return M
            })()
        }
    },
    z => {
        z.O(0, [502], () => z(z.s = 1676)), z.O()
    }
]);